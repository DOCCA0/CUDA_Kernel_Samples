#include <cstdlib>
#include <iostream>
#include <float.h>
#include <algorithm>
#include <cuda_runtime.h>

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
void max_cpu(float* input, float* output, int N) {
    *output =  *(std::max_element(input, input + N));  // 计算输入数组的最大值
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


int main(){
    constexpr size_t N=1280000;
    constexpr int repeat = 10;

    float *input_h, *output_h;
    float *input_d, *output_d;

    input_h = (float*)malloc(N * sizeof(float));
    output_h = (float*)malloc(sizeof(float));
    *output_h = -FLT_MAX;
    for (int i=0;i<N;++i){
        input_h[i]=(float)(i);
    }

    // cpu max
    float total_time_h=TIME_RECORD(repeat, ([&]{max_cpu(input_h, output_h, N);}));
    std::cout<<"CPU max time:"<<total_time_h/repeat<<" ms"<<std::endl;
    std::cout<<"Max value from CPU: "<<*output_h<<std::endl;

    // gpu max
    cudaMalloc(&input_d, N*sizeof(float));
    cudaMalloc(&output_d, sizeof(float));
    cudaMemcpy(input_d, input_h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_d, output_h, sizeof(float), cudaMemcpyHostToDevice);
    constexpr size_t blockSize = 128;
    constexpr size_t gridSize = CEIL(N, blockSize);
    float total_time_d=TIME_RECORD(repeat, ([&]{
        max_kernel<<<gridSize, blockSize>>>(input_d, output_d, N);
    }));
    *output_h = -FLT_MAX;
    cudaMemcpy(output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout<<"GPU max time:"<<total_time_d/repeat<<" ms"<<std::endl;
    std::cout<<"Max value from GPU: "<<*output_h<<std::endl;

    free(input_h);
    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);
    return 0;
}