
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>

// 每个线程负责C矩阵中的一个元素计算，但是全局内存
__global__ __launch_bounds__(1024)
void sgemm_v1(const float *A, const float *B, float *C, int M, int N, int K,float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += alpha * A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum + beta * C[row * N + col];
}

// 每个线程负责C矩阵中的一个元素计算，按照block分块，利用共享内存
template <int BLOCK_SIZE>
__global__ void sgemm_v2(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {
    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    int bx=blockIdx.x;
    int by=blockIdx.y;
    int t_row=threadIdx.x % BN;
    int t_col=threadIdx.x / BN;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float sum = 0.0f;
    // 移动窗口
    for (int k = 0; k < K; k += BK) {
        As[t_row * BK + t_col] = A[t_row * K + k + t_col];
        Bs[t_row * BN + t_col] = B[t_row * N + t_col ];
        __syncthreads();
        for (int n = 0; n < BK; ++n) {
            sum += As[t_row * BK + n] * Bs[n * BN + t_col];
        }
        __syncthreads();
    }
    C[t_row * N + t_col] = sum * alpha + beta * C[t_row * N + t_col];
}