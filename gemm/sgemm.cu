
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>

// v1 每个线程负责C矩阵中的一个元素计算，但是全局内存
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

// v2 每个线程负责C矩阵中的一个元素计算，按照block分块，利用共享内存
template <int BLOCK_SIZE>
__global__ void sgemm_v2(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {
    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    int bx=blockIdx.x;
    int by=blockIdx.y;
    // C在窗口内部的行列
    int row_c=threadIdx.x % BN;
    int col_c=threadIdx.x / BN;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float sum = 0.0f;
    // 移动窗口
    for (int k = 0; k < K; k += BK) {
        As[row_c * BK + col_c] = A[row_c * K + col_c];
        Bs[row_c * BN + col_c] = B[row_c * N + col_c ];
        __syncthreads();
        for (int n = 0; n < BK; ++n) {
            sum += As[row_c * BK + n] * Bs[n * BN + col_c];
        }
        // 移动AB到下一个矩阵块
        A += BK;
        B += BK * N;
        __syncthreads();
    }
    C[row_c * N + col_c] = sum * alpha + beta * C[row_c * N + col_c];
}

// v3 每个线程负责C一列，利用寄存器和共享内存
template <const int BM , const int BN, const int BK, const int TM>
__global__ void sgemm_v3( float *A,  float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int thread_num = BM * BN / TM;

    int col_c = threadIdx.x / BN;
    int row_c = threadIdx.x % BN * TM;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    int col_a =threadIdx.x % BK;
    int row_a = threadIdx.x / BK;
    int stride_a = thread_num / BK;

    int col_b = threadIdx.x % BN;
    int row_b = threadIdx.x / BN;
    int stride_b = thread_num / BN;

    // 额外的一个寄存器用于缓存,tmp[TM]保存的是Bs[row_c * BN + col_c];访存压缩
    float tmp[TM+1] = {0.0f};

    // 移动窗口
    for (int k = 0; k < K; k += BK) {
        for (int i=0;i<BM; i += stride_a) {
            As[(row_a + i) * BK + col_a] = A[(row_a + i) * K + k + col_a];
        }
        for (int i=0;i<BN; i += stride_b) {
            Bs[(row_b + i) * BN + col_b] = B[(row_b + i) * N + k + col_b];
        }
        __syncthreads();
        for (int i=0;i<BK;++i){
            tmp[TM]=Bs[row_c * BN + col_c];
            for (int j=0;j<TM;++j){
                tmp[j] += As[(row_c + j) * BK + i] * tmp[TM];
            }
        }
        // 移动AB到下一个矩阵块
        A += BK;
        B += BK * N;
        __syncthreads();
    }

    for (int i=0;i<TM;++i){
        C[(row_c + i) * N + col_c] = tmp[i] * alpha + beta * C[(row_c + i) * N + col_c];
    }

}