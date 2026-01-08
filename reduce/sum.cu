
#include <cstdlib>
#include <iostream>
#include <float.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <random>

void _cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error]:" << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }
}
#define TIME_RECORD(N, func)                                                                    \
    [&] {                                                                                       \
        float total_time = 0;                                                                   \
        for (int rpt = 0; rpt <= N; ++rpt) {                                           \
            cudaEvent_t start, stop;                                                            \
            _cudaCheck(cudaEventCreate(&start));                                                 \
            _cudaCheck(cudaEventCreate(&stop));                                                  \
            _cudaCheck(cudaEventRecord(start));                                                  \
            cudaEventQuery(start);                                                              \
            func();                                                                             \
            _cudaCheck(cudaEventRecord(stop));                                                   \
            _cudaCheck(cudaEventSynchronize(stop));                                              \
            float elapsed_time;                                                                 \
            _cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));                        \
            if (rpt > 0) total_time += elapsed_time;                                         \
            _cudaCheck(cudaEventDestroy(start));                                                 \
            _cudaCheck(cudaEventDestroy(stop));                                                  \
        }                                                                                       \
        if (N == 0) return (float)0.0;                                                          \
        return total_time;                                                                      \
    }()
void max_cpu(float* input, float* output, int N) {
    *output =  *(std::max_element(input, input + N));  // 计算输入数组的最大值
}

#define CEIL(a, b) ((a + b-1) / (b))



// reduce_v0：使用全局内存，一个线程处理一个元素
__global__ void reduce_v0(float* x_d, float* y_d, int N) {
    // 一个block内负责的起始和结束位置
    const int start = blockIdx.x * blockDim.x;
    const int end = start + blockDim.x;
    const int tid = threadIdx.x;
    const int gid = start + tid;
    
    // block内规约
    for (int offset = blockDim.x>>1;offset>0;offset>>=1){
        if (tid < offset){
            if (gid < N){
                const int partner_gid = gid + offset;
                x_d[gid] += partner_gid < N ?x_d[partner_gid]:0.0f;
            }
        }
        __syncthreads();
    }
    // 第0个thread负责将每个block的结果写入y_d
    if (tid == 0) {
        y_d[blockIdx.x] = x_d[start];
    }

    // grid内规约，后面cpu端处理
}


// reduce_v1：使用静态共享内存，一个线程处理一个元素，但是每个block的数据搬运到共享内存中
template <unsigned int blockSize>
__global__ void reduce_v1(float* x_d, float* y_d, int n) {
    __shared__ float sdata[blockSize];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    // 搬运数据到共享内存
    sdata[tid] = (gid < n) ? x_d[gid] : 0.0f;
    __syncthreads();

    // block内规约
    for ( int offset = blockDim.x>>1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            int partner_tid = tid + offset;
            sdata[tid] += partner_tid<blockDim.x ? sdata[partner_tid] : 0.0f;
        }
        __syncthreads();
    }
    // 第0个thread负责将每个block的结果写入y_d
    if (tid == 0) {
        y_d[blockIdx.x] = sdata[0];
    }
    // grid内规约，后面cpu端处理
}


// reduce_v2：使用动态共享内存，一个线程处理一个元素，但是每个block的数据搬运到共享内存中
__global__ void reduce_v2(float* x_d, float* y_d, int n) {
    extern __shared__ float sdata[];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    // 搬运数据到共享内存
    sdata[tid] = (gid < n) ? x_d[gid] : 0.0f;
    __syncthreads();

    // block内规约
    for ( int offset = blockDim.x>>1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            int partner_tid = tid + offset;
            sdata[tid] += partner_tid<blockDim.x ? sdata[partner_tid] : 0.0f;
        }
        __syncthreads();
    }
    // 第0个thread负责将每个block的结果写入y_d
    if (tid == 0) {
        y_d[blockIdx.x] = sdata[0];
    }
    // grid内规约，后面cpu端处理
}


// reduce_v3: atomicAdd实现Grid内规约
__global__ void reduce_v3(float* x_d, float* y_d, int n) {
    extern __shared__ float sdata[];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    // 搬运数据到共享内存
    sdata[tid] = (gid < n) ? x_d[gid] : 0.0f;
    __syncthreads();

    // block内规约
    for ( int offset = blockDim.x>>1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            int partner_tid = tid + offset;
            sdata[tid] += partner_tid<blockDim.x ? sdata[partner_tid] : 0.0f;
        }
        __syncthreads();
    }
    // grid内规约
    if (tid == 0) {
        // 相当于*d_y_d += sdata[0];
        atomicAdd(y_d, sdata[0]);
    }
    
}

// reduce_v4: warp shuffle实现warp内和block内规约，atomicAdd实现grid内规约
__global__ void reduce_v4(float* x_d, float* y_d, int n) {
    // 32=max(warpSize,warpNum)
    __shared__ float sdata[32];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;

    // warp内规约
    float val = (gid < n) ? x_d[gid] : 0.0f;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();

    // block内规约
    if (warp_id == 0) {
        int warpNum = CEIL(blockDim.x, warpSize);
        val = (lane_id < warpNum) ? sdata[lane_id] : 0.0f;
        for (int offset = warpNum >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        // grid内规约
        if (lane_id == 0) {
            atomicAdd(y_d, val);
        }
    }

}

#define FLOAT4(a) *(float4*)(&a)
// reduce_v5: 使用float4优化内存带宽
__global__ void reduce_v5(float* x_d, float* y_d, int n) {
    // 32=max(warpSize,warpNum)
    __shared__ float sdata[32];
    // 乘以4
    int gid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int tid = threadIdx.x;
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;

    float val=0.0f;
    // float4合并 float val = (gid < n) ? x_d[gid] : 0.0f;
    if(gid + 3 < n){
        float4 tmp = FLOAT4(x_d[gid]);
        val = tmp.x + tmp.y + tmp.z + tmp.w;
    }else{
        for(int i=gid;i<n;i++){
            val+=x_d[i];
        }
    }

    
    // warp内规约
    
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();

    // block内规约
    if (warp_id == 0) {
        int warpNum = CEIL(blockDim.x, warpSize);
        val = (lane_id < warpNum) ? sdata[lane_id] : 0.0f;
        for (int offset = warpNum >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        // grid内规约
        if (lane_id == 0) {
            atomicAdd(y_d, val);
        }
    }

}


template <unsigned int blockSize>
void call_reduce_v0(float* d_nums, float* d_rd_nums, float* h_rd_nums, int N, float* sum) {
    int gridSize = CEIL(N, blockSize);
    reduce_v0<<<gridSize, blockSize>>>(d_nums, d_rd_nums, N);
    cudaMemcpy(h_rd_nums, d_rd_nums, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    *sum = 0;
    for (int i = 0; i < gridSize; i++) {
        *sum += h_rd_nums[i];
    }
}

template <unsigned int blockSize>
void call_reduce_v1(float* d_nums, float* d_rd_nums, float* h_rd_nums, int N, float* sum) {
    int gridSize = CEIL(N, blockSize);
    reduce_v1<blockSize><<<gridSize, blockSize>>>(d_nums, d_rd_nums, N);
    cudaMemcpy(h_rd_nums, d_rd_nums, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    *sum = 0;
    for (int i = 0; i < gridSize; i++) {
        *sum += h_rd_nums[i];
    }
}

template <unsigned int blockSize>
void call_reduce_v2(float* d_nums, float* d_rd_nums, float* h_rd_nums, int N, float* sum) {
    int gridSize = CEIL(N, blockSize);
    reduce_v2<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_nums, d_rd_nums, N);
    cudaMemcpy(h_rd_nums, d_rd_nums, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    *sum = 0;
    for (int i = 0; i < gridSize; i++) {
        *sum += h_rd_nums[i];
    }
}

template <unsigned int blockSize>
void call_reduce_v3(float* d_nums, float* d_rd_nums, float* h_rd_nums, int N, float* sum) {
    int gridSize = CEIL(N, blockSize);
    cudaMemset(d_rd_nums, 0, sizeof(float));
    reduce_v3<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_nums, d_rd_nums, N);
    cudaMemcpy(sum, d_rd_nums, sizeof(float), cudaMemcpyDeviceToHost);
}

template <unsigned int blockSize>
void call_reduce_v4(float* d_nums, float* d_rd_nums, float* h_rd_nums, int N, float* sum) {
    int gridSize = CEIL(N, blockSize);
    cudaMemset(d_rd_nums, 0, sizeof(float));
    reduce_v4<<<gridSize, blockSize>>>(d_nums, d_rd_nums, N);
    cudaMemcpy(sum, d_rd_nums, sizeof(float), cudaMemcpyDeviceToHost);
}

template <unsigned int blockSize>
void call_reduce_v5(float* d_nums, float* d_rd_nums, float* h_rd_nums, int N, float* sum) {
    int gridSize = CEIL(N, blockSize * 4);
    cudaMemset(d_rd_nums, 0, sizeof(float));
    reduce_v5<<<gridSize, blockSize>>>(d_nums, d_rd_nums, N);
    cudaMemcpy(sum, d_rd_nums, sizeof(float), cudaMemcpyDeviceToHost);
}

void randomize_matrix(float *mat, int N) {
    std::random_device rd;  
    std::mt19937 gen(rd()); // 使用随机设备初始化生成器  

    // 创建一个在[0, 2000)之间均匀分布的分布对象  
    std::uniform_int_distribution<> dis(0, 2000); 
    for (int i = 0; i < N; i++) {
        // 生成随机数，限制范围在[-1.0,1.0]
        mat[i] = (dis(gen)-1000)/1000.0;  
    }
}

void host_reduce(float* x, const int N, float* sum) {
    *sum = 0.0;
    for (int i = 0; i < N; i++) {
        *sum += x[i];
    }
}
int main() {
    size_t N = 100000000;
    constexpr size_t BLOCK_SIZE = 128;
    const int repeat_times = 10;

    // 1. host
    float *nums_h = (float *)malloc(sizeof(float) * N);
    float *sum = (float *)malloc(sizeof(float));
    randomize_matrix(nums_h, N);
    
    float total_time_h = TIME_RECORD(repeat_times, ([&]{host_reduce(nums_h, N, sum);}));
    std::cout << "CPU res:" << *sum << ", CPU reduce time: " << total_time_h / repeat_times << " ms" << std::endl;

    // 2. device
    float *d_nums, *d_rd_nums;
    cudaMalloc((void **) &d_nums, sizeof(float) * N);
    cudaMalloc((void **) &d_rd_nums, sizeof(float) * CEIL(N, BLOCK_SIZE));
    float *h_rd_nums = (float *)malloc(sizeof(float) * CEIL(N, BLOCK_SIZE));

    // reduce_v0,全局内存
    cudaMemcpy(d_nums, nums_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    call_reduce_v0<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);
    float sum0 = *sum;

    cudaMemcpy(d_nums, nums_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_0 = TIME_RECORD(repeat_times, ([&]{call_reduce_v0<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);}));
    printf("[reduce_v0]: sum = %f, total_time_0 = %f ms\n", sum0, total_time_0 / repeat_times);

    // reduce_v1
    cudaMemcpy(d_nums, nums_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_1 = TIME_RECORD(repeat_times, ([&]{call_reduce_v1<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);}));
    printf("[reduce_v1]: sum = %f, total_time_1 = %f ms\n", *sum, total_time_1 / repeat_times);

    // reduce_v2
    cudaMemcpy(d_nums, nums_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_2 = TIME_RECORD(repeat_times, ([&]{call_reduce_v2<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);}));
    printf("[reduce_v2]: sum = %f, total_time_2 = %f ms\n", *sum, total_time_2 / repeat_times);

    // reduce_v3
    cudaMemcpy(d_nums, nums_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_3 = TIME_RECORD(repeat_times, ([&]{call_reduce_v3<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);}));
    printf("[reduce_v3]: sum = %f, total_time_3 = %f ms\n", *sum, total_time_3 / repeat_times);

    // reduce_v4
    cudaMemcpy(d_nums, nums_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_4 = TIME_RECORD(repeat_times, ([&]{call_reduce_v4<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);}));
    printf("[reduce_v4]: sum = %f, total_time_4 = %f ms\n", *sum, total_time_4 / repeat_times);

    // reduce_v5
    cudaMemcpy(d_nums, nums_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_5 = TIME_RECORD(repeat_times, ([&]{call_reduce_v5<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);}));
    printf("[reduce_v5]: sum = %f, total_time_5 = %f ms\n", *sum, total_time_5 / repeat_times);
    
    free(nums_h);
    free(sum);
    free(h_rd_nums);
    cudaFree(d_nums);
    cudaFree(d_rd_nums);
}