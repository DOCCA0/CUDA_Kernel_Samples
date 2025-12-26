#include <cstring>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>


#define CEIL(a, b) ((a + b-1) / (b))

void _cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error]:" << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 适合K在[32,128]
__global__ void sgemv_k32(float* A, float* x, float* y, int M,int K) {
    // warp内id
    int lane_id = threadIdx.x % warpSize;
    // 一个block（warp）负责一行M
    int row = blockIdx.x ;

    float res=0.0f;
    // 每个线程处理多个元素
    int kIters = CEIL(K, warpSize);

    #pragma unroll
    for(int i=0; i<kIters; ++i){
        int col = i * warpSize + lane_id;
        res += (col < K) ? A[row * K + col] * x[col] : 0.0f;
    }

    // warp内归约
    for(int offset = warpSize >> 1;offset > 0; offset >>= 1) 
        res += __shfl_down_sync(0xffffffff, res, offset);

    if(lane_id == 0)
        y[row] = res;
}

int main(){
    size_t M=1024;
    size_t K=32;

    float *A_h, *x_h, *y_h;
    float *A_d, *x_d, *y_d,*y_h_cublas;

    A_h = (float*)malloc(M * K * sizeof(float));
    x_h = (float*)malloc(K * sizeof(float));
    y_h = (float*)malloc(M * sizeof(float));
    y_h_cublas = (float*)malloc(M * sizeof(float));

    _cudaCheck(cudaMalloc(&A_d,M*K*sizeof(float)));
    _cudaCheck(cudaMalloc(&x_d,K*sizeof(float)));
    _cudaCheck(cudaMalloc(&y_d,M*sizeof(float)));

    // 初始化A
    for (int i=0;i<M*K;++i){
        A_h[i]=(float)i/K;
    }
    // 初始化x
    for (int i=0;i<K;++i){
        x_h[i]=1.0f;
    }
    // 初始化y
    memset(y_h, 0, M * sizeof(float));
    memset(y_h_cublas, 0, M * sizeof(float));

    cudaEvent_t start, stop;
    _cudaCheck(cudaEventCreate(&start));
    _cudaCheck(cudaEventCreate(&stop));

    float ms= 0.0f;
    int iter=1000;
    _cudaCheck(cudaMemcpy(A_d,A_h,M*K*sizeof(float),cudaMemcpyHostToDevice));
    _cudaCheck(cudaMemcpy(x_d,x_h,K*sizeof(float),cudaMemcpyHostToDevice));
    _cudaCheck(cudaMemcpy(y_d,y_h,M*sizeof(float),cudaMemcpyHostToDevice));


    // 每秒的GFLOP
    double GFLOPS[2] = {0, 0};
    // 总共的GFLOP
    double GFLOPs = 2.0 * M * 1 * K / 1e9;
    double duriations[2] = {0, 0};

    // Warmup
    for (int run=0;run<iter;++run){
        sgemv_k32<<<dim3(M), dim3(32)>>>(A_d, x_d, y_d, M, K);
    }
    
    _cudaCheck(cudaEventRecord(start));
    for (int run=0;run<iter;++run){
        sgemv_k32<<<dim3(M), dim3(32)>>>(A_d, x_d, y_d, M, K);
    }
    _cudaCheck(cudaEventRecord(stop));
    _cudaCheck(cudaEventSynchronize(stop));
    _cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    _cudaCheck(cudaMemcpy(y_h, y_d, M*sizeof(float), cudaMemcpyDeviceToHost));
    
    duriations[0] = ms / iter;
    GFLOPS[0] = GFLOPs / (duriations[0] / 1e3);

    std::cout << "SGEMV K=32, Time: " << duriations[0] << " ms, Performance: " << GFLOPS[0] << " GFLOPS" << std::endl;


    // 用cublas试试
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0;
    float beta = 0;
    _cudaCheck(cudaMemcpy( y_d, y_h, M * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup
    for (int run = 0 ; run < iter; run ++ ) {
        cublasSgemv (handle, CUBLAS_OP_T,
            K, M, &alpha,
            A_d, K, x_d, 1, &beta, y_d, 1
        );
    }

    _cudaCheck(cudaEventRecord(start));
    for (int run = 0 ; run < iter; run ++ ) {
        cublasSgemv (handle, CUBLAS_OP_T,
            K, M, &alpha,
            A_d, K, x_d, 1, &beta, y_d, 1
        );
    }

    _cudaCheck(cudaEventRecord(stop));
    _cudaCheck(cudaEventSynchronize(stop));
    _cudaCheck(cudaEventElapsedTime(&ms, start, stop));

    _cudaCheck(cudaMemcpy( y_h_cublas, y_d, M * sizeof(float), cudaMemcpyDeviceToHost));
    duriations[1] = ms / iter;
    GFLOPS[1] = (GFLOPs ) / (duriations[1] / 1000.0f);
    std::cout << "CUBLAS SGEMV K=32, Time: " << duriations[1] << " ms, Performance: " << GFLOPS[1] << " GFLOPS" << std::endl;

    cublasDestroy(handle);
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M; i++) {
        double rel_err = (y_h[i]-y_h_cublas[i]) / y_h[i] / M;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, y_h[i], y_h[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    
    // Free Memory
    cudaFree(A_d);
    cudaFree(x_d);
    cudaFree(y_d);
    
    free(A_h);
    free(x_h);
    free(y_h);
    free(y_h_cublas);
}