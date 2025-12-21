#include "cuda_runtime.h"
#include "iostream"
// #include <cstdlib>


#define FLOAT4(a) *(float4*)(&a)
#define CEIL(a, b) ((a + b-1) / (b))

void _cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error]:" << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void elementwise_add(float* a, float* b, float* c, int n){
    int idx=(blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >=n)
        return;
    float4 tmp_a = FLOAT4(a[idx]);
    float4 tmp_b = FLOAT4(b[idx]);
    float4 tmp_c;
    tmp_c.x = tmp_a.x + tmp_b.x;
    tmp_c.y = tmp_a.y + tmp_b.y;
    tmp_c.z = tmp_a.z + tmp_b.z;
    tmp_c.w = tmp_a.w + tmp_b.w;
    FLOAT4(c[idx]) = tmp_c;
}

int main(){
    int n=7;
    float *a_h, *b_h, *c_h;
    float *a_d, *b_d, *c_d;

    a_h=(float*)malloc(n*sizeof(float));
    b_h=(float*)malloc(n*sizeof(float));
    c_h=(float*)malloc(n*sizeof(float));
    for (int i=0;i<n;i++){
        a_h[i]=(float)i;
        b_h[i]=(float)(i*10);
    }

    _cudaCheck(cudaMalloc(&a_d,n*sizeof(float)));
    _cudaCheck(cudaMalloc(&b_d,n*sizeof(float)));
    _cudaCheck(cudaMalloc(&c_d,n*sizeof(float)));
    _cudaCheck(cudaMemcpy(a_d,a_h,n*sizeof(float),cudaMemcpyHostToDevice));
    _cudaCheck(cudaMemcpy(b_d,b_h,n*sizeof(float),cudaMemcpyHostToDevice));
    int block_size=1024;
    int grid_size=CEIL(CEIL(n,4),block_size);

    elementwise_add<<<grid_size,block_size>>>(a_d,b_d,c_d,n);
    _cudaCheck(cudaMemcpy(c_h,c_d,n*sizeof(float),cudaMemcpyDeviceToHost));

    for (int i=0;i<n;i++){
        std::cout<<a_h[i]<<" + "<<b_h[i]<<" = "<<c_h[i]<<std::endl;
    }


}