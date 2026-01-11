#include "iostream"
#include <cmath>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include "float.h"
#include "algorithm"

void _cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error]:" << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CEIL(a, b) ((a + b-1) / (b))
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


// reduce_v4: warp shuffle实现warp内和block内规约，atomicAdd实现grid内规约
__global__ void reduce_sum_v4_with_maxval(float* x_d, float* y_d, int n, float* max_val) {
    // 32=max(warpSize,warpNum)
    __shared__ float sdata[32];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;

    // warp内规约 这里减去max_val
    float val = (gid < n) ?expf(x_d[gid] - *max_val) : 0.0f;
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

// __device 代表函数被GPU调用
// static两个作用：1.函数作用域仅限于本文件 2.函数内联展开
__device__ static float atomicMax(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void max_kernel(float *input, float *output, int N){
    // max(warpSize,warpNum)
    __shared__ float sdata[32]; 
    // grid内全局线程id
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // block内warpid
    int warp_id = threadIdx.x / warpSize;
    // block内laneId
    int lane_id = threadIdx.x % warpSize;

    // warp内规约
    float val = (idx < N) ? input[idx] : -FLT_MAX;
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        val =fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    if (lane_id == 0){
        sdata[warp_id] = val;
    }
    __syncthreads();

    // block内规约
    if (warp_id == 0){
        int warpNum=CEIL(blockDim.x, warpSize);
        val = (lane_id < warpNum) ? sdata[lane_id] : -FLT_MAX;
        for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
            val =fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        // grid内规约
        if (lane_id == 0){
            atomicMax(output, val);
        }
    }
}

// 一个thread负责一个softmax
__global__ void softmax_kernel(float* input_list_d, float* output_list_d, int N, float* max_val, float* sum_exp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output_list_d[idx] = expf(input_list_d[idx] - *max_val) / *sum_exp;
    }
}

// 没有减去max_val的softmax
void call_softmax_v1(float* input_list_d, float* output_list_d, float* max_val, float* sum_exp, int N) {
    int blockSize = 256;
    int gridSize = CEIL(N, blockSize);

    // 初始化
    _cudaCheck(cudaMemset(sum_exp, 0, sizeof(float)));
    _cudaCheck(cudaMemset(max_val, 0, sizeof(float)));

    // 计算sum_exp
    reduce_sum_v4_with_maxval<<<gridSize, blockSize>>>(input_list_d, sum_exp, N, max_val);
    
    // 计算softmax
    softmax_kernel<<<gridSize, blockSize>>>(input_list_d, output_list_d, N, max_val, sum_exp);
}

__global__ void setToNegativeMax(float* d_value) {
    *d_value = -FLT_MAX;
}

// 减去max_val的softmax
void call_softmax_v2(float* input_list_d, float* output_list_d, float* max_val, float* sum_exp, int N) {
    int blockSize = 256;
    int gridSize = CEIL(N, blockSize);

    // 初始化
    _cudaCheck(cudaMemset(sum_exp, 0, sizeof(float)));
    // 必须
    setToNegativeMax<<<1, 1>>>(max_val);

    // 计算max_val
    max_kernel<<<gridSize, blockSize>>>(input_list_d, max_val, N);

    // 计算sum_exp
    reduce_sum_v4_with_maxval<<<gridSize, blockSize>>>(input_list_d, sum_exp, N, max_val);
    
    // 计算softmax
    softmax_kernel<<<gridSize, blockSize>>>(input_list_d, output_list_d, N, max_val, sum_exp);
}

void softmax_cpu(float* input, float* output, int N) {
    float max_val_h = *(std::max_element(input, input + N));  // 计算输入数组的最大值
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        output[i] = std::exp(input[i] - max_val_h);  // 每个数先减去最大值，再求exp，避免溢出
        sum += output[i];
    }
    for (int i = 0; i < N; i++) {
        output[i] /= sum;
    }
}

int main(){
    
    const int N = 2048000;
    constexpr size_t BLOCK_SIZE = 256;
    const int repeat_times = 10;
    float* input_list_h, *input_list_d, *output_list_h_cpu,*output_list_h_v1,*output_list_h_v2, *output_list_d, *max_val_h, *max_val_d, *sum_exp_h, *sum_exp_d;
    input_list_h = (float*)malloc(N * sizeof(float));
    output_list_h_cpu = (float*)malloc(N * sizeof(float));
    output_list_h_v1 = (float*)malloc(N * sizeof(float));
    output_list_h_v2 = (float*)malloc(N * sizeof(float));
    max_val_h = (float*)malloc(sizeof(float));
    sum_exp_h = (float*)malloc(sizeof(float));



    for (int i = 0; i < N; ++i) {
        input_list_h[i] = i/(float)N; // 初始化输入数据
    }

    // CPU softmax
    float time_cpu = TIME_RECORD(repeat_times, ([&] {
        softmax_cpu(input_list_h, output_list_h_cpu, N);
    }));
    std::cout << "CPU softmax time: " << time_cpu / repeat_times << " ms" << std::endl;

    
    _cudaCheck(cudaMalloc(&input_list_d, N * sizeof(float)));
    _cudaCheck(cudaMalloc(&output_list_d, N * sizeof(float)));
    _cudaCheck(cudaMalloc(&max_val_d, sizeof(float)));
    _cudaCheck(cudaMalloc(&sum_exp_d, sizeof(float)));
    _cudaCheck(cudaMemcpy(input_list_d, input_list_h, N * sizeof(float), cudaMemcpyHostToDevice));
    // softmax v1
    float time_v1 = TIME_RECORD(repeat_times, ([&] {
        call_softmax_v1(input_list_d, output_list_d, max_val_d, sum_exp_d, N);
    }));
    _cudaCheck(cudaMemcpy(output_list_h_v1, output_list_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "GPU softmax v1 time: " << time_v1 / repeat_times << " ms" << std::endl;

    // softmax v2
    float time_v2 = TIME_RECORD(repeat_times, ([&] {
        call_softmax_v2(input_list_d, output_list_d, max_val_d, sum_exp_d, N);
    }));
    _cudaCheck(cudaMemcpy(output_list_h_v2, output_list_d, N * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "GPU softmax v2 time: " << time_v2 / repeat_times << " ms" << std::endl;

    // 验证结果,对比cpu、v1、v2结果
    float max_diff_v1 = 0.0f;
    float max_diff_v2 = 0.0f;
    for (int i = 0; i < N; ++i) {
        max_diff_v1 = std::max(max_diff_v1, std::abs(output_list_h_cpu[i] - output_list_h_v1[i]));
        max_diff_v2 = std::max(max_diff_v2, std::abs(output_list_h_cpu[i] - output_list_h_v2[i]));
    }
    std::cout << "Max difference between CPU and GPU softmax v1: " << max_diff_v1 << std::endl;
    std::cout << "Max difference between CPU and GPU softmax v2: " << max_diff_v2 << std::endl;

    free(input_list_h);
    free(output_list_h_cpu);
    free(output_list_h_v1);
    free(output_list_h_v2);
    free(max_val_h);
    free(sum_exp_h);
    _cudaCheck(cudaFree(input_list_d));
    _cudaCheck(cudaFree(output_list_d));
    _cudaCheck(cudaFree(max_val_d));
    _cudaCheck(cudaFree(sum_exp_d));    
}