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

    const int x1 = blockIdx.x * TILE_DIM + threadIdx.x;
    const int y1 = blockIdx.y * TILE_DIM + threadIdx.y;

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

    const int x1 = blockIdx.x * TILE_DIM + threadIdx.x;
    const int y1 = blockIdx.y * TILE_DIM + threadIdx.y;

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

    const int x1 = blockIdx.x * TILE_DIM + threadIdx.x;
    const int y1 = blockIdx.y * TILE_DIM + threadIdx.y;

    // 合并读取
    if (x1 < N && y1 < M) {
        tile[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[y1 * N + x1];
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

// =================== main 测试函数 ===================
void randomize_matrix(float* mat, int N) {
    for (int i = 0; i < N; i++) {
        mat[i] = (float)(rand() % 100) / 100.0f;
    }
}

bool verify_matrix(float *mat1, float *mat2, int N) {
    double diff = 0.0;
    for (int i = 0; i < N; i++) {
        diff = std::abs(mat1[i] - mat2[i]);
        if (diff > 0.01) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i],mat2[i],i);
            return false;
        }
    }
    return true;
}

int main() {
    size_t M = 128000;
    size_t N = 1280;
    constexpr size_t BLOCK_SIZE = 32;
    const int repeat_times = 10;

    float *h_matrix = (float *)malloc(sizeof(float) * M * N);
    float *h_matrix_tr_ref = (float *)malloc(sizeof(float) * N * M);
    randomize_matrix(h_matrix, M * N);
    host_transpose(h_matrix, M, N, h_matrix_tr_ref);

    float *d_matrix;
    cudaMalloc((void **) &d_matrix, sizeof(float) * M * N);
    cudaMemcpy(d_matrix, h_matrix, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    free(h_matrix);

    // v0
    float *d_output0;
    cudaMalloc((void **) &d_output0, sizeof(float) * N * M);
    float *h_output0 = (float *)malloc(sizeof(float) * N * M);
    dim3 block_size0(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size0(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
    cudaEvent_t start0, stop0;
    cudaEventCreate(&start0);
    cudaEventCreate(&stop0);
    cudaEventRecord(start0);
    for (int i = 0; i < repeat_times; ++i) {
        device_transpose_v0<<<grid_size0, block_size0>>>(d_matrix, M, N, d_output0);
    }
    cudaEventRecord(stop0);
    cudaEventSynchronize(stop0);
    float ms0 = 0.0f;
    cudaEventElapsedTime(&ms0, start0, stop0);
    cudaMemcpy(h_output0, d_output0, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    verify_matrix(h_output0, h_matrix_tr_ref, N * M);
    printf("[device_transpose_v0] Average time: (%f) ms\n", ms0 / repeat_times);
    cudaFree(d_output0);
    free(h_output0);

    // v1
    float *d_output1;
    cudaMalloc((void **) &d_output1, sizeof(float) * N * M);
    float *h_output1 = (float *)malloc(sizeof(float) * N * M);
    dim3 block_size1(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size1(CEIL(M, BLOCK_SIZE), CEIL(N, BLOCK_SIZE));
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    for (int i = 0; i < repeat_times; ++i) {
        device_transpose_v1<<<grid_size1, block_size1>>>(d_matrix, M, N, d_output1);
    }
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float ms1 = 0.0f;
    cudaEventElapsedTime(&ms1, start1, stop1);
    cudaMemcpy(h_output1, d_output1, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    verify_matrix(h_output1, h_matrix_tr_ref, N * M);
    printf("[device_transpose_v1] Average time: (%f) ms\n", ms1 / repeat_times);
    cudaFree(d_output1);
    free(h_output1);

    // v2
    float *d_output2;
    cudaMalloc((void **) &d_output2, sizeof(float) * N * M);
    float *h_output2 = (float *)malloc(sizeof(float) * N * M);
    dim3 block_size2(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size2(CEIL(M, BLOCK_SIZE), CEIL(N, BLOCK_SIZE));
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    for (int i = 0; i < repeat_times; ++i) {
        device_transpose_v2<<<grid_size2, block_size2>>>(d_matrix, M, N, d_output2);
    }
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float ms2 = 0.0f;
    cudaEventElapsedTime(&ms2, start2, stop2);
    cudaMemcpy(h_output2, d_output2, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    verify_matrix(h_output2, h_matrix_tr_ref, N * M);
    printf("[device_transpose_v2] Average time: (%f) ms\n", ms2 / repeat_times);
    cudaFree(d_output2);
    free(h_output2);

    // v3
    float *d_output3;
    cudaMalloc((void **) &d_output3, sizeof(float) * N * M);
    float *h_output3 = (float *)malloc(sizeof(float) * N * M);
    dim3 block_size3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size3(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3);
    for (int i = 0; i < repeat_times; ++i) {
        device_transpose_v3<BLOCK_SIZE><<<grid_size3, block_size3>>>(d_matrix, M, N, d_output3);
    }
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    float ms3 = 0.0f;
    cudaEventElapsedTime(&ms3, start3, stop3);
    cudaMemcpy(h_output3, d_output3, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    verify_matrix(h_output3, h_matrix_tr_ref, N * M);
    printf("[device_transpose_v3] Average time: (%f) ms\n", ms3 / repeat_times);
    cudaFree(d_output3);
    free(h_output3);

    // v4
    float *d_output4;
    cudaMalloc((void **) &d_output4, sizeof(float) * N * M);
    float *h_output4 = (float *)malloc(sizeof(float) * N * M);
    dim3 block_size4(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size4(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
    cudaEvent_t start4, stop4;
    cudaEventCreate(&start4);
    cudaEventCreate(&stop4);
    cudaEventRecord(start4);
    for (int i = 0; i < repeat_times; ++i) {
        device_transpose_v4<BLOCK_SIZE><<<grid_size4, block_size4>>>(d_matrix, M, N, d_output4);
    }
    cudaEventRecord(stop4);
    cudaEventSynchronize(stop4);
    float ms4 = 0.0f;
    cudaEventElapsedTime(&ms4, start4, stop4);
    cudaMemcpy(h_output4, d_output4, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    verify_matrix(h_output4, h_matrix_tr_ref, N * M);
    printf("[device_transpose_v4] Average time: (%f) ms\n", ms4 / repeat_times);
    cudaFree(d_output4);
    free(h_output4);

    // v5
    float *d_output5;
    cudaMalloc((void **) &d_output5, sizeof(float) * N * M);
    float *h_output5 = (float *)malloc(sizeof(float) * N * M);
    dim3 block_size5(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size5(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
    cudaEvent_t start5, stop5;
    cudaEventCreate(&start5);
    cudaEventCreate(&stop5);
    cudaEventRecord(start5);
    for (int i = 0; i < repeat_times; ++i) {
        device_transpose_v5<BLOCK_SIZE><<<grid_size5, block_size5>>>(d_matrix, M, N, d_output5);
    }
    cudaEventRecord(stop5);
    cudaEventSynchronize(stop5);
    float ms5 = 0.0f;
    cudaEventElapsedTime(&ms5, start5, stop5);
    cudaMemcpy(h_output5, d_output5, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    verify_matrix(h_output5, h_matrix_tr_ref, N * M);
    printf("[device_transpose_v5] Average time: (%f) ms\n", ms5 / repeat_times);
    cudaFree(d_output5);
    free(h_output5);

    free(h_matrix_tr_ref);
    cudaFree(d_matrix);
    return 0;
}