// main.cu
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>
#include <random>

#define CUDA_CHECK(condition)                                          \
  do {                                                                 \
    cudaError_t error = condition;                                     \
    if (error != cudaSuccess) {                                        \
      printf("CUDA_CHECK error in line %d of file %s: %s\n", __LINE__, \
             __FILE__, cudaGetErrorString(cudaGetLastError()));        \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

#define CUBLAS_CHECK(condition)                                            \
  do {                                                                     \
    cublasStatus_t status = condition;                                     \
    if (status != CUBLAS_STATUS_SUCCESS) {                                 \
      printf("CUBLAS_CHECK error in line %d of file %s: %d\n", __LINE__,   \
             __FILE__, status);                                            \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

// #define DEBUG

#ifdef DEBUG
#define DEBUG_BLOCK(expr) \
  do {                    \
    expr                  \
  } while (0)
#else
#define DEBUG_BLOCK(...) \
  do {                   \
  } while (0)
#endif

// -------------------------------
// CUDA Kernels (unchanged)
// -------------------------------

__global__ void naive_nrow_gemm(float *A, float *B, float *C, float a, float b,
                                int M, int N, int K, int mBlock) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx *= mBlock;

  for (int i = idx; i < idx + mBlock; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[j * K + k];
      }
      C[i * N + j] = a * sum + b * C[i * N + j];
    }
  }
}

__global__ void naive_pv(float *P, float *V, float *O, int M, int N,
                         int mBlock) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx *= mBlock;

  int K = M;
  for (int i = idx; i < idx + mBlock; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.f;
      for (int k = 0; k < K; k++) {
        sum += P[i * K + k] * V[k * N + j];
      }
      O[i * N + j] = sum;
    }
  }
}

__global__ void row_softmax(float *input, float *output, int n) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  float max_val = -INFINITY;
  float sum = 0.f;

  for (int i = 0; i < n; i++) {
    if (input[idx * n + i] > max_val) {
      max_val = input[idx * n + i];
    }
  }

  for (int i = 0; i < n; i++) {
    output[idx * n + i] = expf(input[idx * n + i] - max_val);
    sum += output[idx * n + i];
  }

  for (int i = 0; i < n; i++) {
    output[idx * n + i] /= sum;
  }
}

__global__ void transpose(const float *input, float *output, int rows,
                          int cols) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int total = rows * cols;

  if (idx < total) {
    int row = idx / cols;
    int col = idx % cols;
    output[col * rows + row] = input[row * cols + col];
  }
}

void init_random(float *data, size_t num_elements, std::mt19937 &gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < num_elements; i++) {
    data[i] = dist(gen);
  }
}

void self_attention_cuda(float *Q, float *K, float *V, float *O, int m, int n) {
  // 每个线程负责mBlock列的输出
  int mBlock = 2;

  float sm_scale = 1.f / sqrtf(static_cast<float>(n));
  float *sm_o;
  cudaMalloc((void **)&sm_o, sizeof(float) * m * m);

  // Q[M, N] @ K^T[N, M] -> sm_o[M, M]
  dim3 qk_block(m / mBlock, 1, 1);
  naive_nrow_gemm<<<1, qk_block>>>(Q, K, sm_o, sm_scale, 0, m, m, n, mBlock);
  cudaDeviceSynchronize();
  DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError()); printf("== naive QK done ==\n"););

  // sm_o[M, M] -> sm_o[M, M]，每行做softmax
  dim3 sm_block(m, 1, 1);
  row_softmax<<<1, sm_block>>>(sm_o, sm_o, m);
  cudaDeviceSynchronize();
  DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
              printf("== naive softmax(QK) done ==\n"););

  // sm_o[M, M] @ V[M, N] -> O[M, N]，每个线程负责mBlock列的输出
  dim3 qkv_block(m / mBlock, 1, 1);
  naive_pv<<<1, qkv_block>>>(sm_o, V, O, m, n, mBlock);
  cudaDeviceSynchronize();
  DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
              printf("== naive softmax(QK)V done ==\n"););

  cudaFree(sm_o);
}

void cublas_row_major_gemm(cublasHandle_t handle, const float *A,
                           const float *B, float *C, int m, int n, int k,
                           float alpha, float beta) {
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                           B, n, A, k, &beta, C, n));
}

void self_attention_cublas(cublasHandle_t handle, float *Q, float *K, float *V,
                           float *O, int m, int n) {
  float *K_t;
  float *scores;
  float sm_scale = 1.f / sqrtf(static_cast<float>(n));
  float one = 1.f;
  float zero = 0.f;

  CUDA_CHECK(cudaMalloc(&K_t, sizeof(float) * n * m));
  CUDA_CHECK(cudaMalloc(&scores, sizeof(float) * m * m));

  int threads = 256;
  int blocks = (m * n + threads - 1) / threads;
  transpose<<<blocks, threads>>>(K, K_t, m, n);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  cublas_row_major_gemm(handle, Q, K_t, scores, m, m, n, sm_scale, zero);

  dim3 sm_block(m, 1, 1);
  row_softmax<<<1, sm_block>>>(scores, scores, m);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  cublas_row_major_gemm(handle, scores, V, O, m, n, m, one, zero);

  CUDA_CHECK(cudaFree(K_t));
  CUDA_CHECK(cudaFree(scores));
}

bool compare_outputs(const float *actual, const float *expected,
                     size_t num_elements) {
  float max_abs_diff = 0.f;
  double sum_abs_diff = 0.0;

  for (size_t i = 0; i < num_elements; i++) {
    float diff = fabsf(actual[i] - expected[i]);
    max_abs_diff = fmaxf(max_abs_diff, diff);
    sum_abs_diff += diff;
  }

  double mean_abs_diff = sum_abs_diff / static_cast<double>(num_elements);
  printf("Compare with cuBLAS: max_abs_diff=%e, mean_abs_diff=%e\n",
         max_abs_diff, mean_abs_diff);
  return max_abs_diff < 1e-4f;
}

// -------------------------------
// Entry point
// -------------------------------
int main() {
  const int m = 64;
  const int n = 128;
  size_t num_elements = m * n;

  // Host memory
  float *h_Q = new float[num_elements];
  float *h_K = new float[num_elements];
  float *h_V = new float[num_elements];
  float *h_O = new float[num_elements];
  float *h_cublas_O = new float[num_elements];

  std::mt19937 gen(42);
  init_random(h_Q, num_elements, gen);
  init_random(h_K, num_elements, gen);
  init_random(h_V, num_elements, gen);

  // Device memory
  float *d_Q, *d_K, *d_V, *d_O, *d_cublas_O;
  CUDA_CHECK(cudaMalloc(&d_Q, num_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_K, num_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_V, num_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_O, num_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cublas_O, num_elements * sizeof(float)));

  // Copy to device
  CUDA_CHECK(cudaMemcpy(d_Q, h_Q, num_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, h_K, num_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_V, h_V, num_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  printf("Running self-attention for m=%d, n=%d\n", m, n);
  self_attention_cuda(d_Q, d_K, d_V, d_O, m, n);

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  self_attention_cublas(handle, d_Q, d_K, d_V, d_cublas_O, m, n);
  CUBLAS_CHECK(cublasDestroy(handle));

  // Copy result back
  CUDA_CHECK(cudaMemcpy(h_O, d_O, num_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_cublas_O, d_cublas_O, num_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));

  bool passed = compare_outputs(h_O, h_cublas_O, num_elements);
  printf("Validation %s\n", passed ? "PASSED" : "FAILED");

  // Cleanup
  delete[] h_Q;
  delete[] h_K;
  delete[] h_V;
  delete[] h_O;
  delete[] h_cublas_O;
  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_O);
  cudaFree(d_cublas_O);

  return 0;
}
