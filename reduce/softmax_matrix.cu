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


    
// cpu计算每行softmax
void softmax_row_cpu(float* input, float* output,int M, int N) {
    for (int row = 0; row < M; ++row) {
        float* row_input = input + row * N;
        float* row_output = output + row * N;
        float max_val = *(std::max_element(row_input, row_input + N));  // 计算输入数组的最大值
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            row_output[i] = std::exp(row_input[i] - max_val);  // 每个数先减去最大值，再求exp，避免溢出
            sum += row_output[i];
        }
        for (int i = 0; i < N; i++) {
            row_output[i] /= sum;
        }
    }

}

// cpu计算每列softmax
void softmax_col_cpu(float* input, float* output,int M, int N) {
    for(int col = 0; col < N; ++col){
        float* col_input = input + col;
        float* col_output = output + col;
        float max_val = -FLT_MAX;
        for(int row = 0; row < M; ++row){
            if(col_input[row * N] > max_val)
                max_val = col_input[row * N];
        }
        float sum = 0.0f;
        for(int row = 0; row < M; ++row){
            col_output[row * N] = std::exp(col_input[row * N] - max_val);
            sum += col_output[row * N];
        }
        for(int row = 0; row < M; ++row){
            col_output[row * N] /= sum;
        }
    }
}

// gpu计算每行softmax,一个block处理一行
__global__ void softmax_row_kernel(float* input_matrix, float* output_matrix, int M, int N) {
    __shared__ float shared_max_row[32];
    __shared__ float shared_sum_row[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int row = blockIdx.x;
    if (row >= M) return;
    // 每个block要处理多少次
    int iter = CEIL(N, blockDim.x);

    // 1.求行最大
    float thread_warp_max_val = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < iter; ++i) {
        int col = i * blockDim.x + threadIdx.x;
        // 1.1 计算每个线程的最大值
        thread_warp_max_val = (col < N) ? fmaxf(thread_warp_max_val, input_matrix[row * N + col]) : thread_warp_max_val;
    }
    // 1.2 warp内归约求最大值
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        thread_warp_max_val = fmaxf(thread_warp_max_val, __shfl_down_sync(0xffffffff, thread_warp_max_val, offset));
    }
    if (lane_id == 0) {
        shared_max_row[warp_id] = thread_warp_max_val;
    }
    __syncthreads();

    // 1.3 block内归约求最大值
    float block_max_val = -FLT_MAX;
    if (warp_id == 0) {
        int warpNum = CEIL(blockDim.x, warpSize);
        block_max_val = (lane_id < warpNum) ? shared_max_row[lane_id] : -FLT_MAX;
        #pragma unroll
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            block_max_val = fmaxf(block_max_val, __shfl_down_sync(0xffffffff, block_max_val, offset));
        }
        if (lane_id == 0) {
            // 这里shared_max_row[0]相当于tmp
            shared_max_row[0] = block_max_val;
        }
    }
    __syncthreads();

    // !!广播，如果使用的__shfl_xor_sync则不需要这一步
    block_max_val = shared_max_row[0];

    // 2.求行和
    float thread_warp_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < iter; ++i) {
        int col = i * blockDim.x + threadIdx.x;;
        // 2.1 计算每个线程的和
        thread_warp_sum += (col < N) ? expf(input_matrix[row * N + col] - block_max_val) : 0.0f;
    }
    // 2.2 warp内归约求和
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        thread_warp_sum += __shfl_down_sync(0xffffffff, thread_warp_sum, offset);
    }
    if (lane_id == 0) {
        shared_sum_row[warp_id] = thread_warp_sum;
    }
    __syncthreads();
    // 2.3 block内归约求和
    float block_sum = 0.0f;
    if (warp_id == 0) {
        int warpNum = CEIL(blockDim.x, warpSize);
        block_sum = (lane_id < warpNum) ? shared_sum_row[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane_id == 0) {
            // 这里shared_sum_row[0]相当于tmp
            shared_sum_row[0] = block_sum;
        }
    }
    __syncthreads();
    block_sum = shared_sum_row[0];

    // 3.计算softmax
    #pragma unroll
    for (int i = 0; i < iter; ++i) {
        int col = i * blockDim.x + threadIdx.x;
        if (col < N) {
            output_matrix[row * N + col] = expf(input_matrix[row * N + col] - block_max_val) / block_sum;
        }
    }
}


// gpu计算每列softmax,一个block处理一列
__global__ void softmax_col_kernel(float* input_matrix, float* output_matrix, int M, int N) {
    __shared__ float shared_max_col[32];
    __shared__ float shared_sum_col[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int col = blockIdx.x;
    if (col >= N) return;
    // 每个block要处理多少次
    int iter = CEIL(M, blockDim.x);

    // 1.求列最大
    float thread_warp_max_val = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < iter; ++i) {
        int row = i * blockDim.x + threadIdx.x;
        // 1.1 计算每个线程的最大值
        thread_warp_max_val = (row < M) ? fmaxf(thread_warp_max_val, input_matrix[row * N + col]) : thread_warp_max_val;
    }
    // 1.2 warp内归约求最大值
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        thread_warp_max_val = fmaxf(thread_warp_max_val, __shfl_down_sync(0xffffffff, thread_warp_max_val, offset));
    }
    if (lane_id == 0) {
        shared_max_col[warp_id] = thread_warp_max_val;
    }
    __syncthreads();

    // 1.3 block内归约求最大值
    float block_max_val = -FLT_MAX;
    if (warp_id == 0) {
        int warpNum = CEIL(blockDim.x, warpSize);
        block_max_val = (lane_id < warpNum) ? shared_max_col[lane_id] : -FLT_MAX;
        #pragma unroll
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            block_max_val = fmaxf(block_max_val, __shfl_down_sync(0xffffffff, block_max_val, offset));
        }
        if (lane_id == 0) {
            // 这里shared_max_col[0]相当于tmp
            shared_max_col[0] = block_max_val;
        }
    }
    __syncthreads();
    block_max_val = shared_max_col[0];
    // 2.求列和
    float thread_warp_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < iter; ++i) {
        int row = i * blockDim.x + threadIdx.x;
        // 2.1 计算每个线程的和
        thread_warp_sum += (row < M) ? expf(input_matrix[row * N + col] - block_max_val) : 0.0f;
    }
    // 2.2 warp内归约求和
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        thread_warp_sum += __shfl_down_sync(0xffffffff, thread_warp_sum, offset);
    }
    if (lane_id == 0) {
        shared_sum_col[warp_id] = thread_warp_sum;
    }
    __syncthreads();

    // 2.3 block内归约求和
    float block_sum_val = 0.0f;
    if (warp_id == 0) {
        int warpNum = CEIL(blockDim.x, warpSize);
        block_sum_val = (lane_id < warpNum) ? shared_sum_col[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            block_sum_val += __shfl_down_sync(0xffffffff, block_sum_val, offset);
        }
        if (lane_id == 0) {
            // 这里shared_sum_col[0]相当于tmp
            shared_sum_col[0] = block_sum_val;
        }
    }
    __syncthreads();
    block_sum_val = shared_sum_col[0];
    // 3.计算softmax
    #pragma unroll
    for (int i = 0; i < iter; ++i) {
        int row = i * blockDim.x + threadIdx.x;
        if (row < M) {
            output_matrix[row * N + col] = expf(input_matrix[row * N + col] - block_max_val) / block_sum_val;
        }
    }
}



// 行softmax，一个block等于一个warp处理一行，使用xor省略广播步骤
__global__ void __launch_bounds__(32) softmax_row_kernel_warp(float* input_matrix, float* output_matrix, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    int iter = CEIL(N, 32);
    int lane_id = threadIdx.x; 

    // ================= Phase 1: 求最大值 (Max) =================
    float val_max = -FLT_MAX;
    
    // 1.1 线程私有归约 (处理 N > 32 的情况)
    for (int i = 0; i < iter; ++i) {
        int col = i * 32 + lane_id;
        if (col < N) {
            val_max = fmaxf(val_max, input_matrix[row * N + col]);
        }
    }

    // 1.2 Warp 内全员归约 (Butterfly All-Reduce)
    // 循环结束后，Warp内所有线程的寄存器 val_max 都是全局最大值
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xffffffff, val_max, offset);
        val_max = fmaxf(val_max, other);
    }
    // 【优势】不需要写入 Shared Memory，也不需要广播，所有人都拿到了最大值

    // ================= Phase 2: 求指数和 (Sum) =================
    float val_sum = 0.0f;

    // 2.1 线程私有求和
    for (int i = 0; i < iter; ++i) {
        int col = i * 32 + lane_id;
        if (col < N) {
            val_sum += expf(input_matrix[row * N + col] - val_max);
        }
    }

    // 2.2 Warp 内全员归约 (Butterfly All-Reduce)
    // 循环结束后，所有线程的 val_sum 都是全局总和
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xffffffff, val_sum, offset);
        val_sum += other;
    }

    // ================= Phase 3: 计算并写回 =================
    // 此时每个线程都有正确的 val_max 和 val_sum，直接计算即可
    for (int i = 0; i < iter; ++i) {
        int col = i * 32 + lane_id;
        if (col < N) {
            output_matrix[row * N + col] = expf(input_matrix[row * N + col] - val_max) / val_sum;
        }
    }
}


// 列softmax，一个block等于一个warp处理一列，使用xor省略广播步骤
__global__ void __launch_bounds__(32) softmax_col_kernel_warp(float* input_matrix, float* output_matrix, int M, int N) {
    int col = blockIdx.x;
    if (col >= N) return;   
    int iter = CEIL(M, 32);
    int lane_id = threadIdx.x;
    // ================= Phase 1: 求最大值 (Max) =================
    float val_max = -FLT_MAX;
    // 1.1 线程私有归约 (处理 M > 32 的情况)
    for (int i = 0; i < iter; ++i) {
        int row = i * 32 + lane_id;
        if (row < M) {
            val_max = fmaxf(val_max, input_matrix[row * N + col]);
        }
    }

    // 1.2 Warp 内全员归约 (Butterfly All-Reduce)
    // 循环结束后，所有线程的 val_max 都是全局最大值
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xffffffff, val_max, offset);
        val_max = fmaxf(val_max, other);
    }
    // 【优势】不需要写入 Shared Memory，也不需要广播，所有人都拿到了最大值

    // ================= Phase 2: 求指数和 (Sum) =================
    float val_sum = 0.0f;

    // 2.1 线程私有求和
    for (int i = 0; i < iter; ++i) {
        int row = i * 32 + lane_id;
        if (row < M) {
            val_sum += expf(input_matrix[row * N + col] - val_max);
        }
    }

    // 2.2 Warp 内全员归约 (Butterfly All-Reduce)
    // 循环结束后，所有线程的 val_sum 都是全局总和
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xffffffff, val_sum, offset);
        val_sum += other;
    }

    // ================= Phase 3: 计算并写回 =================
    // 此时每个线程都有正确的 val_max 和 val_sum，直接计算即可
    for (int i = 0; i < iter; ++i) {
        int row = i * 32 + lane_id;
        if (row < M) {
            output_matrix[row * N + col] = expf(input_matrix[row * N + col] - val_max) / val_sum;
        }
    }
}

int main() {
    const int M = 20480;
    const int N = 640;
    const int repeat_times = 10;
    
    size_t size = M * N * sizeof(float);
    float *h_input = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);
    float *h_output_gpu = (float*)malloc(size);

    srand(time(0));
    for (int i = 0; i < M * N; ++i) {
        h_input[i] = (float)rand() / RAND_MAX;
    }

    float *d_input, *d_output;
    _cudaCheck(cudaMalloc(&d_input, size));
    _cudaCheck(cudaMalloc(&d_output, size));

    _cudaCheck(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // ==========================================
    // Row Softmax
    // ==========================================
    printf("================ Row Softmax ================\n");
    
    // CPU
    float cpu_time = TIME_RECORD(repeat_times, [&] {
        softmax_row_cpu(h_input, h_output_cpu, M, N);
    });
    printf("CPU time: %f ms\n", cpu_time / repeat_times);

    // GPU Kernel 1 (Block)
    dim3 block(128);
    dim3 grid(M);
    float gpu_time = TIME_RECORD(repeat_times, ([&] {
        softmax_row_kernel<<<grid, block>>>(d_input, d_output, M, N);
    }));
    _cudaCheck(cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for(int i=0; i<M*N; ++i) {
        if(fabs(h_output_cpu[i] - h_output_gpu[i]) > 1e-4) {
             printf("Row Kernel 1 Error at %d: cpu=%f, gpu=%f\n", i, h_output_cpu[i], h_output_gpu[i]);
             correct = false;
             break;
        }
    }
    if(correct) printf("Row Kernel 1 (Block) time: %f ms, Result: PASS\n", gpu_time / repeat_times);

    // GPU Kernel 2 (Warp)
    dim3 block_warp(32);
    dim3 grid_warp(M);
    gpu_time = TIME_RECORD(repeat_times, ([&] {
        softmax_row_kernel_warp<<<grid_warp, block_warp>>>(d_input, d_output, M, N);
    }));
    _cudaCheck(cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost));
    
    correct = true;
    for(int i=0; i<M*N; ++i) {
        if(fabs(h_output_cpu[i] - h_output_gpu[i]) > 1e-4) {
             printf("Row Kernel 2 Error at %d: cpu=%f, gpu=%f\n", i, h_output_cpu[i], h_output_gpu[i]);
             correct = false;
             break;
        }
    }
    if(correct) printf("Row Kernel 2 (Warp) time: %f ms, Result: PASS\n", gpu_time / repeat_times);


    // ==========================================
    // Col Softmax
    // ==========================================
    printf("\n================ Col Softmax ================\n");

    // CPU
    // Reset output for safety
    memset(h_output_cpu, 0, size);
    
    cpu_time = TIME_RECORD(repeat_times, [&] {
        softmax_col_cpu(h_input, h_output_cpu, M, N);
    });
    printf("CPU time: %f ms\n", cpu_time / repeat_times);

    // GPU Kernel 1 (Block)
    dim3 block_col(128);
    dim3 grid_col(N);
    gpu_time = TIME_RECORD(repeat_times, ([&] {
        softmax_col_kernel<<<grid_col, block_col>>>(d_input, d_output, M, N);
    }));
    _cudaCheck(cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost));
    
    correct = true;
    for(int i=0; i<M*N; ++i) {
        if(fabs(h_output_cpu[i] - h_output_gpu[i]) > 1e-4) {
             printf("Col Kernel 1 Error at %d: cpu=%f, gpu=%f\n", i, h_output_cpu[i], h_output_gpu[i]);
             correct = false;
             break;
        }
    }
    if(correct) printf("Col Kernel 1 (Block) time: %f ms, Result: PASS\n", gpu_time / repeat_times);

    // GPU Kernel 2 (Warp)
    dim3 block_col_warp(32);
    dim3 grid_col_warp(N);
    gpu_time = TIME_RECORD(repeat_times, ([&] {
        softmax_col_kernel_warp<<<grid_col_warp, block_col_warp>>>(d_input, d_output, M, N);
    }));
    _cudaCheck(cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost));
    
    correct = true;
    for(int i=0; i<M*N; ++i) {
        if(fabs(h_output_cpu[i] - h_output_gpu[i]) > 1e-4) {
             printf("Col Kernel 2 Error at %d: cpu=%f, gpu=%f\n", i, h_output_cpu[i], h_output_gpu[i]);
             correct = false;
             break;
        }
    }
    if(correct) printf("Col Kernel 2 (Warp) time: %f ms, Result: PASS\n", gpu_time / repeat_times);


    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);

    return 0;
}


