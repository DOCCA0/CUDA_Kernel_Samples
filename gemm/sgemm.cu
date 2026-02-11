
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <vector_types.h>

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

    // 额外的一个寄存器用于缓存,tmp[TM]保存的是Bs[row_c * BN][col_c];访存压缩
    float tmp[TM+1] = {0.0f};

    // 移动窗口
    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for (int i=0;i<BM; i += stride_a) {
            As[(row_a + i) * BK][col_a] = A[(row_a + i) * K  + col_a];
        }
        #pragma unroll
        for (int i=0;i<BN; i += stride_b) {
            Bs[(row_b + i) * BN][col_b] = B[(row_b + i) * N + col_b];
        }
        __syncthreads();
        #pragma unroll
        for (int i=0;i<BK;++i){
            tmp[TM]=Bs[i * BN][col_c];
            #pragma unroll
            for (int j=0;j<TM;++j){
                tmp[j] += As[(row_c + j) * BK][i] * tmp[TM];
            }
        }
        // 移动AB到下一个矩阵块
        A += BK;
        B += BK * N;
        __syncthreads();
    }

    #pragma unroll
    for (int i=0;i<TM;++i){
        C[(row_c + i) * N + col_c] = tmp[i] * alpha + beta * C[(row_c + i) * N + col_c];
    }
}



/**
v3 (1D Tile): 读取 TM 个 A 的元素，读取 1 个 B 的元素。
v4 (2D Tile): 读取 TM 个 A 的元素，读取 TN 个 B 的元素。
代价：为了做 TM * TN 次计算，只需要读取 TM+TN 次共享内存。
*/

template <const int BM , const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v4( float *A,  float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int bx = blockIdx.x;
    int by = blockIdx.y;


    int thread_num = BM * BN / (TM * TN);

    // 一个线程负责计算 TM * TN 个 C 矩阵元素
    // 一共需要 BM/TM * BN/TN 个线程
    // (row_c, col_c) 是 C 矩阵块内的左上角起始位置
    int row_c = threadIdx.x % (BN/TN) * TN;
    int col_c = threadIdx.x / (BN/TN) * TM ;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // (row_a,col_a) 是线程负责的 A 矩阵块内的起始位置
    int col_a =threadIdx.x % BK;
    int row_a = threadIdx.x / BK;
    int stride_a = thread_num / BK;

    // (row_b,col_b) 是线程负责的 B 矩阵块内的起始位置
    int col_b = threadIdx.x % BN;
    int row_b = threadIdx.x / BN;
    int stride_b = thread_num / BN;

    // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值
    float tmp[TM][TN]= {0.f};

    // 移动窗口
    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        // As上列向移动
        #pragma unroll
        for (int i=0;i<BM; i += stride_a) {
            As[(row_a + i) * BK][col_a] = A[(row_a + i) * K  + col_a];
        }
        // Bs上行向移动
        #pragma unroll
        for (int i=0;i<BN; i += stride_b) {
            Bs[(row_b + i) * BN][col_b] = B[(row_b + i) * N + col_b];
        }
        __syncthreads();
        #pragma unroll
        for (int i=0;i<BK;++i){
            #pragma unroll
            for (int j=0;j<TM;++j){
                for (int l=0;l<TN;++l){
                    tmp[j][l] += As[(row_c + j) * BK][i] * Bs[i * BN][col_c + l];
                }
            }
        }
        // 移动AB到下一个矩阵块
        A += BK;
        B += BK * N;
        __syncthreads();
    }

    #pragma unroll
    for (int i=0;i<TM;++i){
        for (int j =0;j<TN;++j){ 
            C[(row_c + i) * N + col_c + j] = alpha * tmp[i][j] + beta * C[(row_c + i) * N + col_c + j];
        }
    }
}


// v5 使用寄存器，减少共享内存访问冲突，每轮 Shared Memory 读取次数 TM * TN * 2 变成了 TM + TN
template <const int BM , const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v5( float *A,  float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int bx = blockIdx.x;
    int by = blockIdx.y 


    int thread_num = BM * BN / (TM * TN);

    // 一个线程负责计算 TM * TN 个 C 矩阵元素
    // 一共需要 BM/TM * BN/TN 个线程
    // (row_c, col_c) 是 C 矩阵块内的左上角起始位置
    int col_c = (threadIdx.x % (BN/TN)) * TN;
    int row_c = (threadIdx.x / (BN/TN)) * TM ;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // (row_a,col_a) 是线程负责的 A 矩阵块内的起始位置
    int col_a =threadIdx.x % BK;
    int row_a = threadIdx.x / BK;
    int stride_a = thread_num / BK;

    // (row_b,col_b) 是线程负责的 B 矩阵块内的起始位置
    int col_b = threadIdx.x % BN;
    int row_b = threadIdx.x / BN;
    int stride_b = thread_num / BN;

    // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值
    float tmp[TM][TN]= {0.f};
    float a_reg[TM];
    float b_reg[TN];
    // 移动窗口
    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        // As上列向移动
        #pragma unroll
        for (int i=0;i<BM; i += stride_a) {
            As[(row_a + i) * BK][col_a] = A[(row_a + i) * K  + col_a];
        }
        // Bs上行向移动
        #pragma unroll
        for (int i=0;i<BN; i += stride_b) {
            Bs[(row_b + i) * BN][col_b] = B[(row_b + i) * N + col_b];
        }
        __syncthreads();
        #pragma unroll
        for (int i=0;i<BK;++i){
            #pragma unroll
            for (int j=0;j<TM;++j){
                a_reg[j] = As[(row_c + j) * BK][i];
            }
            #pragma unroll
            for (int l=0;l<TN;++l){
                b_reg[l] = Bs[i * BN][col_c + l];
            }
            #pragma unroll
            for (int j=0;j<TM;++j){
                for (int l=0;l<TN;++l){
                    tmp[j][l] += a_reg[j] * b_reg[l];
                }
            }
        }
        // 移动AB到下一个矩阵块
        A += BK;
        B += BK * N;
        __syncthreads();
    }

    #pragma unroll
    for (int i=0;i<TM;++i){
        for (int j =0;j<TN;++j){ 
            C[(row_c + i) * N + col_c + j] = alpha * tmp[i][j] + beta * C[(row_c + i) * N + col_c + j];
        }
    }
}


// v6 float4 版本
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
template <const int BM , const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v6( float *A,  float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int bx = blockIdx.x;
    int by = blockIdx.y;


    int thread_num = BM * BN / (TM * TN);

    // 一个线程负责计算 TM * TN 个 C 矩阵元素
    // 一共需要 BM/TM * BN/TN 个线程
    // (row_c, col_c) 是 C 矩阵块内的左上角起始位置
    int col_c = (threadIdx.x % (BN/TN)) * TN;
    int row_c = (threadIdx.x / (BN/TN)) * TM ;

    // As 转置，为了float4读取列
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // (row_a,col_a) 是线程负责的 A 矩阵块内的起始位置
    int col_a =(threadIdx.x % (BK/4)) * 4;
    int row_a = threadIdx.x / (BK/4);
    // stride 是需要跳过的行数(转置后为列)
    int stride_a = thread_num/(BK/4);

    // (row_b,col_b) 是线程负责的 B 矩阵块内的起始位置
    int col_b = (threadIdx.x % (BN/4)) * 4;
    int row_b = threadIdx.x / (BN/4);
    int stride_b = thread_num / (BN/4);

    // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值
    float tmp[TM][TN]= {0.f};
    float a_reg[TM];
    float b_reg[TN];

    // // ldg_a_reg的大小为 每个A的元素大小/线程数（每个线程负责多少个A元素搬运）
    // float ldg_a_reg[BM*BK/thread_num]= {0.f};

    // 移动窗口
    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        // As上列向移动
        #pragma unroll
        for (int i=0;i<BM; i += stride_a) {
            float4 fl4 =FETCH_FLOAT4(A[(row_a + i) * K + col_a]);
            // As转置存，fl4中间缓存，减少共享内存访问冲突
            As[(col_a)][ i + row_a] = fl4.x;
            As[(col_a + 1)][ i + row_a] = fl4.y;
            As[(col_a + 2)][ i + row_a] = fl4.z;
            As[(col_a + 3)][ i + row_a] = fl4.w;
        }
        // Bs上行向移动
        #pragma unroll
        for (int i=0;i<BK; i += stride_b) {
            FETCH_FLOAT4(Bs[row_b + i][col_b]) = FETCH_FLOAT4(B[(row_b + i) * N + col_b]); // 不需要转置
        }
        __syncthreads();
        #pragma unroll
        for (int i=0;i<BK;++i){
            #pragma unroll
            for (int m=0;m<TM;m+=4){
                // 转置就是为了这里
                FETCH_FLOAT4(a_reg[m]) = FETCH_FLOAT4(As[i * BM][(row_c + m) ]);
            }
            #pragma unroll
            for (int n=0;n<TN;n+=4){
                FETCH_FLOAT4(b_reg[n]) = FETCH_FLOAT4(Bs[i * BN][col_c + n]);
            }
            #pragma unroll
            for (int m=0;m<TM;++m){
                for (int n=0;n<TN;++n){
                    tmp[m][n] += a_reg[m] * b_reg[n];
                }
            }
        }
        // 移动AB到下一个矩阵块
        A += BK;
        B += BK * N;
        __syncthreads();
    }

    #pragma unroll
    for (int m=0;m<TM;++m){
        for (int n =0;n<TN;++n){ 
            float4  c_fl4 =FETCH_FLOAT4(C[(row_c + m) * N + col_c + n]);
            c_fl4.x = alpha * tmp[m][n] + beta * c_fl4.x;
            c_fl4.y = alpha * tmp[m][n] + beta * c_fl4.y;
            c_fl4.z = alpha * tmp[m][n] + beta * c_fl4.z;
            c_fl4.w = alpha * tmp[m][n] + beta * c_fl4.w;
            FETCH_FLOAT4(C[(row_c + m) * N + col_c + n]) = c_fl4;
        }
    }
}


// v7 预取下一轮数据到寄存器，计算和访存重叠
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
template <const int BM , const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_v7( float *A,  float *B, float *C, int M, int N, int K, float alpha, float beta) {
    int bx = blockIdx.x;
    int by = blockIdx.y;


    int thread_num = BM * BN / (TM * TN);

    // 一个线程负责计算 TM * TN 个 C 矩阵元素
    // 一共需要 BM/TM * BN/TN 个线程
    // (row_c, col_c) 是 C 矩阵块内的左上角起始位置
    int col_c = (threadIdx.x % (BN/TN)) * TN;
    int row_c = (threadIdx.x / (BN/TN)) * TM ;

    // As 转置，为了float4读取列
    // 增加一倍用于缓存
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // (row_a,col_a) 是线程负责的 A 矩阵块内的起始位置
    int col_a =(threadIdx.x % (BK/4)) * 4;
    int row_a = threadIdx.x / (BK/4);
    // stride 是需要跳过的行数(转置后为列)
    int stride_a = thread_num/(BK/4);

    // (row_b,col_b) 是线程负责的 B 矩阵块内的起始位置
    int col_b = (threadIdx.x % (BN/4)) * 4;
    int row_b = threadIdx.x / (BN/4);
    int stride_b = thread_num / (BN/4);

    // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值
    float tmp[TM][TN]= {0.f};
    float ldg_a_reg[BK * BM / thread_num] = {0.}; // 每个线程负责多少个As的搬运
    float ldg_b_reg[BK * BN / thread_num] = {0.}; // 每个线程负责多少个Bs的搬运

    float a_reg[2][TM];
    float b_reg[2][TN];

    // 预取第一轮数据
    #pragma unroll
    for (int i=0;i<BM; i += stride_a) {
        // 0 4 8 12 ... 
        int ldg_index = i / stride_a * 4;  // 第ldg_index轮
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[(row_a + i) * K + col_a]);
        // As转置存，fl4中间缓存，减少共享内存访问冲突
        As[0][(col_a)][ i + row_a] = ldg_a_reg[ldg_index];
        As[0][(col_a + 1)][ i + row_a] = ldg_a_reg[ldg_index+1];
        As[0][(col_a + 2)][ i + row_a] = ldg_a_reg[ldg_index+2];
        As[0][(col_a + 3)][ i + row_a] = ldg_a_reg[ldg_index+3];
    }
    #pragma unroll
    for (int i=0;i<BK; i += stride_b) {
        FETCH_FLOAT4(Bs[0][row_b + i][col_b]) = FETCH_FLOAT4(B[(row_b + i) * N + col_b]); // 不需要转置
    }
    int write_index = 1;
    int load_index;
    int k=0;
    do {
        __syncthreads();
        k+=BK;
        if(k<K){
            #pragma unroll
            for (int i=0;i<BM; i += stride_a) {
                int ldg_index = i / stride_a * 4;  // 第ldg_index轮
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[(row_a + i) * K + col_a + k]);
            }
            #pragma unroll
            for (int i=0;i<BK; i += stride_b) {
                int ldg_index = i / stride_b * 4;  // 第ldg_index轮
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[(row_b + i) * N + col_b + k]);
            }
        }
        load_index = write_index^1;
        #pragma unroll
        for (int m=0;m<TM;m+=4){
            // 转置就是为了这里
            FETCH_FLOAT4(a_reg[0][m]) = FETCH_FLOAT4(As[load_index][0 * BM][(row_c + m) ]);
        }
        #pragma unroll
        for (int n=0;n<TN;n+=4){
            FETCH_FLOAT4(b_reg[0][n]) = FETCH_FLOAT4(Bs[load_index][0 * BN][col_c + n]);
        }

        // 加载下一次迭代的数据
        #pragma unroll
        for(int bk=0;bk<BK-1;++bk){
            for (int m=0;m<TM;m+=4){
                // 转置就是为了这里
                FETCH_FLOAT4(a_reg[(bk+1)%2][m]) = FETCH_FLOAT4(As[load_index][(bk+1) * BM][(row_c + m) ]);
            }
            #pragma unroll
            for (int n=0;n<TN;n+=4){
                FETCH_FLOAT4(b_reg[(bk+1)%2][n]) = FETCH_FLOAT4(Bs[load_index][(bk+1) * BN][col_c + n]);
            }
            #pragma unroll
            for (int m=0;m<TM;++m){
                for (int n=0;n<TN;++n){
                    tmp[m][n] += a_reg[bk%2][m] * b_reg[bk%2][n];
                }
            }
        }
        // 前面只算了BK-1轮，最后一轮在这里算
        #pragma unroll
        for (int m=0;m<TM;++m){
            for (int n=0;n<TN;++n){
                tmp[m][n] += a_reg[(BK-1)%2][m] * b_reg[(BK-1)%2][n];
            }
        }

        // __syncthreads();  // 这里不需要同步了，因为上面的As(Bs)[load_index]和下面的As(Bs)[write_index]的内存是分开的
        if (k < K) {
            #pragma unroll
            for (int i=0;i<BM; i += stride_a) {
                // 0 4 8 12 ... 
                int ldg_index = i / stride_a * 4;  // 第ldg_index轮
                // As转置存，fl4中间缓存，减少共享内存访问冲突
                As[0][(col_a)][ i + row_a] = ldg_a_reg[ldg_index];
                As[0][(col_a + 1)][ i + row_a] = ldg_a_reg[ldg_index+1];
                As[0][(col_a + 2)][ i + row_a] = ldg_a_reg[ldg_index+2];
                As[0][(col_a + 3)][ i + row_a] = ldg_a_reg[ldg_index+3];
            }
            #pragma unroll
            for (int i=0;i<BK; i += stride_b) {
                // 0 4 8 12 ... 
                int ldg_index = i / stride_b * 4;  // 第ldg_index轮
                FETCH_FLOAT4(Bs[write_index][row_b + i][col_b]) = FETCH_FLOAT4(B[(row_b + i) * N + col_b]); // 不需要转置
            }
        }
        write_index = load_index^1;
    }while (k < K);


    // C=alpha*A*B + beta*C
    #pragma unroll
    for (int m=0;m<TM;++m){
        #pragma unroll
        for (int n =0;n<TN;++n){ 
            float4  c_fl4 =FETCH_FLOAT4(C[(row_c + m) * N + col_c + n]);
            c_fl4.x = alpha * tmp[m][n] + beta * c_fl4.x;
            c_fl4.y = alpha * tmp[m][n] + beta * c_fl4.y;
            c_fl4.z = alpha * tmp[m][n] + beta * c_fl4.z;
            c_fl4.w = alpha * tmp[m][n] + beta * c_fl4.w;
            FETCH_FLOAT4(C[(row_c + m) * N + col_c + n]) = c_fl4;
        }
    }
}