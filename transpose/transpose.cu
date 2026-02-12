#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define CEIL(a, b) ((a + b-1) / (b))

void _cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error]:" << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }
}

void host_transpose(float *input, int M, int N, float *output) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            output[j * M + i] = input[i * N + j];
        }
    }
}


// v0 朴素实现，一个线程负责一个输入元素
__global__ void device_transpose_v0(float *input, int M, int N, float *output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}

// v1 合并写入，一个线程负责一个输出元素（因为写入比读取更耗时）
__global__ void device_transpose_v1(float *input, int M, int N, float *output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        output[row * M + col] = input[col* N + row];
    }
}

// v2 显式调用__ldg（开启只读缓存），减少没有合并读取的影响
__global__ void device_transpose_v2(float *input, int M, int N, float *output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        output[row * M + col] = __ldg(&input[col* N + row]);
    }
}

// v3 使用共享内存，合并读取+写入，但有bank conflict
// 不同的block的bank0指的是同一个bank0，bank的数量永远是32
template <const int TILE_DIM>
__global__ void device_transpose_v3(float *input, int M, int N, float *output) {
    __shared__ float tile[TILE_DIM][TILE_DIM ];

    const int x1 = blockIdx.y * TILE_DIM + threadIdx.y;
    const int y1 = blockIdx.x * TILE_DIM + threadIdx.x;

    // 合并读取
    if (x1 < N && y1 < M) {
        tile[threadIdx.y][threadIdx.x] = input[y1 * N + x1];
    }

    __syncthreads();

    const int x2 = blockIdx.y * TILE_DIM + threadIdx.x;
    const int y2 = blockIdx.x * TILE_DIM + threadIdx.y;
    if (x2 < M && y2 < N) {
        // bank conflict，访问tile[0][0]和tile[32][0]会访问同一个bank0，导致冲突
        // 一个bank中有32个数据，导致了32路冲突
        output[y2 * M + x2] = tile[threadIdx.x][threadIdx.y];
    };
}

// v4 使用共享内存，合并读取+写入，padding解决bank conflict
template <const int TILE_DIM>
__global__ void device_transpose_v4(float *input, int M, int N, float *output) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    const int x1 = blockIdx.y * TILE_DIM + threadIdx.y;
    const int y1 = blockIdx.x * TILE_DIM + threadIdx.x;

    // 合并读取
    if (x1 < N && y1 < M) {
        tile[threadIdx.y][threadIdx.x] = input[y1 * N + x1];
    }

    __syncthreads();

    const int x2 = blockIdx.y * TILE_DIM + threadIdx.x;
    const int y2 = blockIdx.x * TILE_DIM + threadIdx.y;
    if (x2 < M && y2 < N) {
        // 不同的线程访问不同的bank，解决bank conflict
        output[y2 * M + x2] = tile[threadIdx.x][threadIdx.y];
    };
}


// v5 使用共享内存，合并读取+写入，使用使用swizzling解决bank conflict
template <const int TILE_DIM>
__global__ void device_transpose_v5(float *input, int M, int N, float *output) {
    __shared__ float tile[TILE_DIM][TILE_DIM ];

    const int x1 = blockIdx.y * TILE_DIM + threadIdx.y;
    const int y1 = blockIdx.x * TILE_DIM + threadIdx.x;

    // 合并读取
    if (x1 < N && y1 < M) {
        tile[threadIdx.y][threadIdx.x] = input[y1 * N + x1];
    }

    __syncthreads();

    const int x2 = blockIdx.y * TILE_DIM + threadIdx.x;
    const int y2 = blockIdx.x * TILE_DIM + threadIdx.y;
    if (x2 < M && y2 < N) {
        // swizzling主要利用了异或运算的以下两个性质来规避bank conflict：
        // 封闭性：对于任意整数 x、y，x ^ y 的结果仍然是一个整数，并且在 0 到最大线程索引范围内，不会越界。
        // 唯一性：对于固定的 y，x1 ^ y != x2 ^ y 当且仅当 x1 != x2。
        // 0,1,2,3...
        // 1,0,3,2...
        // 2,3,0,1...
        // 3,2,1,0...
        output[y2 * M + x2] = tile[threadIdx.x][threadIdx.x^threadIdx.y];
    };
}