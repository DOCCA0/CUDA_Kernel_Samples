#include <__clang_cuda_builtin_vars.h>
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



// reduce_v1：使用静态共享内存
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