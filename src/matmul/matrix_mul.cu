#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "matrix_mul.h"
using namespace std;

void matrix_mul(float* A, float* B, float* C, int N){
    for(int i = 0; i < N; i++){
	    for(int j = 0; j < N; j++){
	        C[i*N+j] = 0.0f;
	        for(int k = 0; k < N; k++){
	            C[i*N+j] += A[i*N+k] * B[k*N+j];
	        }
	    }
    }
}

__global__ void matmul_kernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

//Tiled Matrix-Matrix Multiplication
__global__ void matmul_kernel_tiling(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float A_shared[TILE_DIM][TILE_DIM];
    __shared__ float B_shared[TILE_DIM][TILE_DIM];
    float sum = 0.0f;
    assert(row < N);
    assert(col < N);
    for(int i = 0; i < N / TILE_DIM; i++){
        A_shared[threadIdx.y][threadIdx.x] = A[row*N+i*TILE_DIM+threadIdx.x];
        B_shared[threadIdx.y][threadIdx.x] = B[(i*TILE_DIM+threadIdx.y)*N+col];
        __syncthreads();

        for(int j = 0; j < TILE_DIM; j++){
            sum += A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x];
        }
        __syncthreads();
    }

    // TODO: If N is not an integer multiple of TILE_DIM, the remaining part needs to be handled.
    C[row * N + col] = sum;
}
__global__ void matmul_kernel_tiling_1(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    __shared__ float A_shared[TILE_DIM][TILE_DIM];
    __shared__ float B_shared[TILE_DIM][TILE_DIM];
    float sum = 0.0f;

    for (int i = 0; i < (N + TILE_DIM - 1) / TILE_DIM; i++) {
        int tiledRow = row;
        int tiledCol = i * TILE_DIM + threadIdx.x;
        A_shared[threadIdx.y][threadIdx.x] = (tiledRow < N && tiledCol < N) ? A[tiledRow * N + tiledCol] : 0.0f;

        tiledRow = i * TILE_DIM + threadIdx.y;
        tiledCol = col;
        B_shared[threadIdx.y][threadIdx.x] = (tiledRow < N && tiledCol < N) ? B[tiledRow * N + tiledCol] : 0.0f;

        __syncthreads();

        for (int j = 0; j < TILE_DIM; j++) {
            sum += A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}


//Tiled and prefetch Matrix-Matrix Multiplication
__global__ void matmul_kernel_tiling_prefetch_2(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N || threadIdx.x >= TILE_DIM || threadIdx.y >= TILE_DIM) return;

    __shared__ float A_shared[2][TILE_DIM][TILE_DIM];
    __shared__ float B_shared[2][TILE_DIM][TILE_DIM];

    float sum = 0.0f;
    int commit_id = 0;

    __pipeline_memcpy_async(&A_shared[1][threadIdx.y][threadIdx.x], &A[row * N + 1 * TILE_DIM + threadIdx.x], sizeof(float));
    __pipeline_memcpy_async(&B_shared[1][threadIdx.y][threadIdx.x], &B[(1 * TILE_DIM + threadIdx.y) * N + col], sizeof(float));
    __pipeline_commit();
    commit_id++;

    A_shared[0][threadIdx.y][threadIdx.x] = A[row * N + 0 * TILE_DIM + threadIdx.x];
    B_shared[0][threadIdx.y][threadIdx.x] = B[(0 * TILE_DIM + threadIdx.y) * N + col];
    int flag = 0;
    __syncthreads();

    for (int i = 0; i < N / TILE_DIM; i++) {
        for (int j = 0; j < TILE_DIM; j++) {
            sum += A_shared[flag][threadIdx.y][j] * B_shared[flag][j][threadIdx.x];
        }
        __syncthreads();

        if(i+2 < N/TILE_DIM){
            __pipeline_memcpy_async(&A_shared[flag][threadIdx.y][threadIdx.x], &A[row * N + (i + 2) * TILE_DIM + threadIdx.x], sizeof(float));
            __pipeline_memcpy_async(&B_shared[flag][threadIdx.y][threadIdx.x], &B[((i + 2) * TILE_DIM + threadIdx.y) * N + col], sizeof(float));
            __pipeline_commit();
            commit_id++;
        }

        __pipeline_wait_prior(commit_id-2);
        flag ^= 1;
        __syncthreads();
    }
    C[row * N + col] = sum;
}

// double buffer for prefetch
__global__ void matmul_kernel_tiling_prefetch(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_shared[2][TILE_DIM][TILE_DIM];
    __shared__ float B_shared[2][TILE_DIM][TILE_DIM];

    float sum = 0.0f;
    int flag = 0;
    //if (row >= N || col >= N || threadIdx.x >= TILE_DIM || threadIdx.y >= TILE_DIM) return;

    A_shared[0][threadIdx.y][threadIdx.x] = A[row * N + 0 * TILE_DIM + threadIdx.x];
    B_shared[0][threadIdx.y][threadIdx.x] = B[(0 * TILE_DIM + threadIdx.y) * N + col];
    __syncthreads();

    for (int i = 0; i < N / TILE_DIM; i++) {
        int next = flag ^ 1;
        if ((i + 1) < N / TILE_DIM) {
            //printf("kernel: hello world\n");
            __pipeline_memcpy_async(&A_shared[next][threadIdx.y][threadIdx.x], &A[row * N + (i+1)*TILE_DIM + threadIdx.x], sizeof(float));
            //printf("kernel:(%d,%d,%f,%d)\n",row, col, A[row * N + (i + 1) * TILE_DIM] + threadIdx.x, ((i + 1) * TILE_DIM + threadIdx.y) * N + col);
            __pipeline_memcpy_async(&B_shared[next][threadIdx.y][threadIdx.x], &B[((i+1)*TILE_DIM + threadIdx.y) * N + col], sizeof(float));
            __pipeline_commit();
        }

        for (int j = 0; j < TILE_DIM; ++j) {
            sum += A_shared[flag][threadIdx.y][j] * B_shared[flag][j][threadIdx.x];
            //printf("[%d]:(%f,%f)\t", j, A_shared[threadIdx.y][k], B_shared[j][threadIdx.x]);
        }

        __pipeline_wait_prior(0);
        __syncthreads();
        flag ^= 1;

    }
   C[row * N + col] = sum;
}
// flag represent different flow
__global__ void matmul_kernel_tiling_prefetch_1(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N || threadIdx.x >= TILE_DIM || threadIdx.y >= TILE_DIM) return;

    __shared__ float A_shared[TILE_DIM][TILE_DIM];
    __shared__ float B_shared[TILE_DIM][TILE_DIM];

    float sum = 0.0f;
    __shared__ float A_shared_next[TILE_DIM][TILE_DIM];
    __shared__ float B_shared_next[TILE_DIM][TILE_DIM];

    A_shared[threadIdx.y][threadIdx.x] = A[row * N + 0 * TILE_DIM + threadIdx.x];
    B_shared[threadIdx.y][threadIdx.x] = B[(0 * TILE_DIM + threadIdx.y) * N + col];
    __syncthreads();

    int flag = 0;

    for (int i = 0; i < N / TILE_DIM; i++) {
        //prefetch next tile
        if ((i + 1) < N / TILE_DIM) {
            if(flag == 0){
                __pipeline_memcpy_async(&A_shared_next[threadIdx.y][threadIdx.x], &A[row * N + (i + 1) * TILE_DIM + threadIdx.x], sizeof(float));
                //printf("kernel:(%d,%d,%d,%d)\n",row, col, row * N + (i + 1) * TILE_DIM + threadIdx.x, ((i + 1) * TILE_DIM + threadIdx.y) * N + col);
                __pipeline_memcpy_async(&B_shared_next[threadIdx.y][threadIdx.x], &B[((i + 1) * TILE_DIM + threadIdx.y) * N + col], sizeof(float));
                __pipeline_commit();
                //printf("kernel: hello world\n");
            }
            else{
                __pipeline_memcpy_async(&A_shared[threadIdx.y][threadIdx.x], &A[row * N + (i + 1) * TILE_DIM + threadIdx.x], sizeof(float));
                __pipeline_memcpy_async(&B_shared[threadIdx.y][threadIdx.x], &B[((i + 1) * TILE_DIM + threadIdx.y) * N + col], sizeof(float));
                __pipeline_commit();
            }
        }
        for (int j = 0; j < TILE_DIM; j++) {
            if(flag == 0) sum += A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x];
            else sum += A_shared_next[threadIdx.y][j] * B_shared_next[j][threadIdx.x];
        }

       // __syncthreads();
        __pipeline_wait_prior(0);
        __syncthreads();
        flag ^= 1;
    }
    C[row * N + col] = sum;
}


//Tiled, prefetch, bank conflict Matrix-Matrix Multiplication
// shared memory conflict. degrate the performance. Maybe the reason is index calculation increase more latency than free conflict
// Todo: what is the type of situation where the random index for bank conflict is useful?
__global__ void matmul_kernel_tiling_prefetch_bank(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float A_shared[TILE_DIM][TILE_DIM];
    __shared__ float B_shared[TILE_DIM][TILE_DIM];
    float sum = 0.0f;
    float A_prefetch = A[row*N+threadIdx.x];
    float B_prefetch = B[threadIdx.y*N+col];
    for(int i = 0; i < N / TILE_DIM; i++){
        A_shared[threadIdx.y][threadIdx.x] = A_prefetch;
        B_shared[threadIdx.y][threadIdx.y ^ threadIdx.x] = B_prefetch;
        __syncthreads();

        if((i+1) < N/TILE_DIM){
            __pipeline_memcpy_async(&A_prefetch, &A[row*N+(i+1)*TILE_DIM+threadIdx.x], sizeof(float));
            __pipeline_memcpy_async(&B_prefetch, &B[((i+1)*TILE_DIM+threadIdx.y)*N+col], sizeof(float));
            __pipeline_commit();
        }

        for(int j = 0; j < TILE_DIM; j++){
            sum += A_shared[threadIdx.y][j] * B_shared[j][j^threadIdx.x];
        }
        __syncthreads();
        __pipeline_wait_prior(0);
    }

    // TODO: If N is not an integer multiple of TILE_DIM, the remaining part needs to be handled.
    C[row * N + col] = sum;
}

__global__ void matmul_kernel_tiling_bank(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float A_shared[TILE_DIM][TILE_DIM];
    __shared__ float B_shared[TILE_DIM][TILE_DIM];
    float sum = 0.0f;
    assert(row < N);
    assert(col < N);
    for(int i = 0; i < N / TILE_DIM; i++){
        A_shared[threadIdx.y][threadIdx.x] = A[row*N+i*TILE_DIM+threadIdx.x];
        B_shared[threadIdx.y][threadIdx.y^threadIdx.x] = B[(i*TILE_DIM+threadIdx.y)*N+col];
        __syncthreads();

        for(int j = 0; j < TILE_DIM; j++){
            sum += A_shared[threadIdx.y][j] * B_shared[j][j^threadIdx.x];
        }
        __syncthreads();
    }

    // TODO: If N is not an integer multiple of TILE_DIM, the remaining part needs to be handled.
    C[row * N + col] = sum;
}

//register Matrix-Matrix Multiplication
//block_dim : tile_dim = 1:1
__global__ void matmul_kernel_tiling_prefetch_register(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float A_shared[TILE_DIM][TILE_DIM];
    __shared__ float B_shared[TILE_DIM][TILE_DIM];
    float sum = 0.0f;
    float A_prefetch = A[row*N+threadIdx.x];
    float B_prefetch = B[threadIdx.y*N+col];
    float B_register[TILE_DIM];
    float A_reg;

    for(int i = 0; i < N / TILE_DIM; i++){
        A_shared[threadIdx.y][threadIdx.x] = A_prefetch;
        B_shared[threadIdx.y][threadIdx.x] = B_prefetch;
        __syncthreads();

        if((i+1) < N/TILE_DIM){
            __pipeline_memcpy_async(&A_prefetch, &A[row*N+(i+1)*TILE_DIM+threadIdx.x], sizeof(float));
            __pipeline_memcpy_async(&B_prefetch, &B[((i+1)*TILE_DIM+threadIdx.y)*N+col], sizeof(float));
            __pipeline_commit();
        }

        #pragma unroll
        for(int j = 0; j < TILE_DIM; j++){
            // A_register[j] = A_shared[threadIdx.x][j];
            B_register[j] = B_shared[j][threadIdx.x];
        }
        __syncthreads();

        A_reg = A_shared[threadIdx.y][0];
        #pragma unroll
        for(int j = 0; j < TILE_DIM-1; j++){
            A_reg = A_shared[threadIdx.y][j+1];
            sum += A_reg * B_register[j];
        }

        sum += A_reg * B_register[TILE_DIM-1];
        __syncthreads();
        __pipeline_wait_prior(0);
    }

    C[row * N + col] = sum;
}

// block_dim : tile_dim = 1:2. each thread process multi elements.
__global__ void matmul_kernel_2tiling_1(float* A, float* B, float* C, int N) {
    int row = 2 * blockIdx.y * blockDim.y + threadIdx.y;
    int col = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float A_shared[TILE_DIM][TILE_DIM];
    __shared__ float B_shared[TILE_DIM][TILE_DIM];
    float sum = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    int stride = TILE_DIM / 2;

    for(int i = 0; i < N / TILE_DIM; i++){
        A_shared[threadIdx.y][threadIdx.x] = A[row*N+i*TILE_DIM+threadIdx.x];
        A_shared[threadIdx.y][stride+threadIdx.x] = A[row*N+i*TILE_DIM+stride+threadIdx.x];
        A_shared[threadIdx.y+stride][threadIdx.x] = A[(row+stride)*N+i*TILE_DIM+threadIdx.x];
        A_shared[threadIdx.y+stride][threadIdx.x+stride] = A[(row+stride)*N+i*TILE_DIM+stride+threadIdx.x];

        B_shared[threadIdx.y][threadIdx.x] = B[(i*TILE_DIM+threadIdx.y)*N+col];
        B_shared[threadIdx.y][stride+threadIdx.x] = B[(i*TILE_DIM+threadIdx.y)*N+stride+col];
        B_shared[stride+threadIdx.y][threadIdx.x] = B[(i*TILE_DIM+threadIdx.y+stride)*N+col];
        B_shared[stride+threadIdx.y][stride+threadIdx.x] = B[(i*TILE_DIM+threadIdx.y+stride)*N+stride+col];

        __syncthreads();

        for(int j = 0; j < TILE_DIM; j++){
            sum += A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x];
            sum1 += A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x+stride];
            sum2 += A_shared[threadIdx.y+stride][j] * B_shared[j][threadIdx.x];
            sum3 += A_shared[threadIdx.y+stride][j] * B_shared[j][threadIdx.x+stride];
        }
        __syncthreads();
    }

    C[row*N + col] = sum;
    C[row*N+stride+col] = sum1;
    C[(row+stride)*N+col] = sum2;
    C[(row+stride)*N+col+stride]  = sum3;
}

__global__ void matmul_kernel_2tiling(float* A, float* B, float* C, int N) {
    int row = 2 * blockIdx.y * blockDim.y + threadIdx.y;
    int col = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float A_shared[TILE_DIM * TILE_DIM];
    __shared__ float B_shared[TILE_DIM * TILE_DIM];
    
    float sum[TILE_FACTOR * TILE_FACTOR];
    #pragma unroll
    for (int i = 0; i < TILE_FACTOR * TILE_FACTOR; i++) {
        sum[i] = 0.0f;
    }


    int stride = TILE_DIM / 2;

    for(int i = 0; i < N / TILE_DIM; i++){
        int a_index, b_index, a_shar_index, b_shar_index;
        #pragma unroll
        for (int j = 0; j < TILE_FACTOR; j++){
            #pragma unroll
            for (int k = 0; k < TILE_FACTOR; k++){
                a_shar_index = (threadIdx.y + j * stride) * TILE_DIM + threadIdx.x + k * stride;
                b_shar_index = a_shar_index;
                a_index = (row + j * stride) * N + i * TILE_DIM + k * stride + threadIdx.x;
                b_index = (i * TILE_DIM + threadIdx.y + stride * j) * N + k * stride + col;

                A_shared[a_shar_index] = A[a_index];
                B_shared[b_shar_index] = B[b_index];
            }
        }

        __syncthreads();

        for(int j = 0; j < TILE_DIM; j++){
            int a_shar_index, b_shar_index, sum_index;
            #pragma unroll
            for (int z = 0; z < TILE_FACTOR; z++){
                #pragma unroll
                for (int k = 0; k < TILE_FACTOR; k++){
                    sum_index = z * TILE_FACTOR + k;
                    a_shar_index = (threadIdx.y + z * stride) * TILE_DIM + j;
                    b_shar_index = j * TILE_DIM + threadIdx.x + k * stride;

                    sum[sum_index] += A_shared[a_shar_index] * B_shared[b_shar_index];
                }
            }
        }
        __syncthreads();
    }

    int sum_index, result_index;
    #pragma unroll
    for (int i = 0; i < TILE_FACTOR; i++){
        #pragma unroll
        for (int j = 0; j < TILE_FACTOR; j++){
            result_index = (row + i * stride) * N + col + j * stride;
            sum_index = i * TILE_FACTOR + j;

            C[result_index] = sum[sum_index];
        }
    }
}

__global__ void matmul_kernel_2tiling_register(float* A, float* B, float* C, int N) {
    int row = 2 * blockIdx.y * blockDim.y + threadIdx.y;
    int col = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float A_shared[TILE_DIM * TILE_DIM];
    __shared__ float B_shared[TILE_DIM * TILE_DIM];
    
    __shared__ float C_shared[TILE_DIM * TILE_DIM];
    float sum[REG_DIM * REG_DIM];

    int stride = TILE_DIM / 2;

    for(int i = 0; i < N / TILE_DIM; i++){
        int a_index, b_index, a_shar_index, b_shar_index;
        #pragma unroll
        for (int j = 0; j < TILE_FACTOR; j++){
            #pragma unroll
            for (int k = 0; k < TILE_FACTOR; k++){
                a_shar_index = (threadIdx.y + j * stride) * TILE_DIM + threadIdx.x + k * stride;
                b_shar_index = a_shar_index;
                a_index = (row + j * stride) * N + i * TILE_DIM + k * stride + threadIdx.x;
                b_index = (i * TILE_DIM + threadIdx.y + stride * j) * N + k * stride + col;

                A_shared[a_shar_index] = A[a_index];
                B_shared[b_shar_index] = B[b_index];
            }
        }

        __syncthreads();

        for(int j = 0; j < TILE_DIM; j++){
            int a_shar_index, b_shar_index, sum_index;
            #pragma unroll
            for (int z = 0; z < TILE_FACTOR; z++){
                #pragma unroll
                for (int k = 0; k < TILE_FACTOR; k++){
                    sum_index = z * TILE_FACTOR + k;
                    a_shar_index = (threadIdx.y + z * stride) * TILE_DIM + j;
                    b_shar_index = j * TILE_DIM + threadIdx.x + k * stride;

                    sum[sum_index] += A_shared[a_shar_index] * B_shared[b_shar_index];
                }
            }
        }
        float A_reg[REG_DIM][REG_DIM], B_reg[REG_DIM][REG_DIM];
        int iter_time = TILE_DIM / REG_DIM;

        #pragma unroll
        for (int i = 0; i < iter_time; i++) {
            #pragma unroll
            for (int j = 0; j < iter_time; j++) {
                // row and col model have two type
                int row_in_tile, col_in_tile;
                row_in_tile = ;
                col_in_tile = ;
                // initial sum tile
                #pragma unroll
                for (int k = 0; k < REG_DIM * REG_DIM; k++) {
                    sum[k] = 0.0f;
                }

                // calculate sum tile 
                int sum_index, a_index, b_index;
                #pragma unroll
                for (int x = 0; x < TILE_DIM / REG_DIM; x++) {
                    for (int y = 0; y < REG_DIM; y++) {
                        for (int z = 0; z < REG_DIM; z++) {
                            sum_index = y * REG_DIM + z;
                            a_index = 
                            b_index = 
                            sum[sum_index] += A_shared[a_index] * B_shared[b_index];
                        }
                    }
                }

                // write back to result
                for (int x = 0; x < REG_DIM; x++){
                    for (int y = 0; y < REG_DIM; y++) {

                        c_index = row * TILE_DIM + col;
                        sum_index = x * REG_DIM + y;
                        C_shared[c_index] = sum[sum_index];
                    }
                    
                }
            }
        }

        __syncthreads();
    }

    int sum_index, result_index;
    #pragma unroll
    for (int i = 0; i < TILE_FACTOR; i++){
        #pragma unroll
        for (int j = 0; j < TILE_FACTOR; j++){
            result_index = (row + i * stride) * N + col + j * stride;
            sum_index = i * TILE_FACTOR + j;

            C[result_index] = sum[sum_index];
        }
    }
}
__global__ void matmul_kernel_2tiling_register_1(float* A, float* B, float* C, int N) {
    int row = 2 * blockIdx.y * blockDim.y + threadIdx.y;
    int col = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float A_shared[TILE_DIM][TILE_DIM];
    __shared__ float B_shared[TILE_DIM][TILE_DIM];
    float sum = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float A_reg[REG_DIM][REG_DIM], B_reg[REG_DIM][REG_DIM];

    int stride = TILE_DIM / 2;

    for(int i = 0; i < N / TILE_DIM; i++){
        A_shared[threadIdx.y][threadIdx.x] = A[row*N+i*TILE_DIM+threadIdx.x];
        A_shared[threadIdx.y][stride+threadIdx.x] = A[row*N+i*TILE_DIM+stride+threadIdx.x];
        A_shared[threadIdx.y+stride][threadIdx.x] = A[(row+stride)*N+i*TILE_DIM+threadIdx.x];
        A_shared[threadIdx.y+stride][threadIdx.x+stride] = A[(row+stride)*N+i*TILE_DIM+stride+threadIdx.x];

        B_shared[threadIdx.y][threadIdx.x] = B[(i*TILE_DIM+threadIdx.y)*N+col];
        B_shared[threadIdx.y][stride+threadIdx.x] = B[(i*TILE_DIM+threadIdx.y)*N+stride+col];
        B_shared[stride+threadIdx.y][threadIdx.x] = B[(i*TILE_DIM+threadIdx.y+stride)*N+col];
        B_shared[stride+threadIdx.y][stride+threadIdx.x] = B[(i*TILE_DIM+threadIdx.y+stride)*N+stride+col];

        __syncthreads();

        for(int j = 0; j < TILE_DIM/REG_DIM; j++){
            A_reg[0][0] = A_shared[threadIdx.y][j];
            A_reg[0][1] = A_shared[threadIdx.y][stride+j];
            A_reg[1][0] = A_shared[stride+threadIdx.y][j];
            A_reg[1][1] = A_shared[stride+threadIdx.y][stride+j];

            B_reg[0][0] = B_shared[j][threadIdx.x];
            B_reg[0][1] = B_shared[j][stride+threadIdx.x];
            B_reg[1][0] = B_shared[stride+j][threadIdx.x];
            B_reg[1][1] = B_shared[stride+j][stride+threadIdx.x];

            sum += A_reg[0][0] * B_reg[0][0] + A_reg[0][1] * B_reg[1][0];
            sum1 += A_reg[0][0] * B_reg[0][1] + A_reg[0][1] * B_reg[1][1];
            sum2 += A_reg[1][0] * B_reg[0][0] + A_reg[1][1] * B_reg[1][0];
            sum3 += A_reg[1][0] * B_reg[0][1] + A_reg[1][1] * B_reg[1][1];
        }
        __syncthreads();
    }

    C[row*N + col] = sum;
    C[row*N+stride+col] = sum1;
    C[(row+stride)*N+col] = sum2;
    C[(row+stride)*N+col+stride]  = sum3;
}
//Todo: register, hardware(tile size, block size), tensor core, device-host data transfer, collaberative.
void launch_matmul(float* d_A, float* d_B, float* d_C, int N) {
    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
    //dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    std::cout << "Launching kernel..." << std::endl;
    //matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    //matmul_kernel_tiling<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    //matmul_kernel_tiling_prefetch<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    //matmul_kernel_tiling_prefetch_bank<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    //matmul_kernel_tiling_bank<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    //matmul_kernel_tiling_prefetch_register<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    matmul_kernel_2tiling<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    //matmul_kernel_2tiling_register<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaError_t err = cudaGetLastError();
    std::cout << "Kernel launch: " << cudaGetErrorString(err) << std::endl;
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    std::cout << "After sync: " << cudaGetErrorString(err) << std::endl;
}

void fill_arry(float* A, float* B, int N){
    for(int i = 0; i < N; i++){
        for (int j = 0; j < N; j++) {
            A[i*N+j] = static_cast<float>(rand() % 5) + 1;
            B[i*N+j] = static_cast<float>(rand() % 5) + 1;
            // A[i*N+j] = 1;
            // B[i*N+j] = static_cast<float>(rand() % 5) + 1;;

        }
    }
}
void test_hello(){
    printf("hello world\n");
}

void print_gpu_info(){
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Found " << deviceCount << " CUDA device(s).\n";

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        std::cout << "\nDevice " << dev << ": " << prop.name << "\n";
        std::cout << "  Compute capabilithreadIdx.y:        " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total global memory:       " << (prop.totalGlobalMem >> 20) << " MB\n";
        std::cout << "  Shared memory per block:   " << prop.sharedMemPerBlock << " bytes\n";
        std::cout << "  Warp size:                 " << prop.warpSize << "\n";
        std::cout << "  Max threads per block:     " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max threads per SM:        " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Multiprocessor count:      " << prop.multiProcessorCount << "\n";
        std::cout << "  Max threads dim (x,y,z):   ("
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << ")\n";
        std::cout << "  Max grid size (x,y,z):     ("
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << ")\n";
        std::cout << "  Memory clock rate (KHz):   " << prop.memoryClockRate << "\n";
        std::cout << "  Memory bus width (bits):   " << prop.memoryBusWidth << "\n";
        std::cout << "  Clock rate (KHz):          " << prop.clockRate << "\n";
        std::cout << "  L2 cache size:             " << prop.l2CacheSize << "\n";
    }

}
void test_matrix_mul_cpu(){
    int row, col;
    row = MATRIX_SIZE;
    col = MATRIX_SIZE;
    float* mat1 = new float[MATRIX_SIZE*MATRIX_SIZE];
    float* mat2 = new float[MATRIX_SIZE*MATRIX_SIZE];
    float* result = new float[MATRIX_SIZE*MATRIX_SIZE];
    //mat1[0] = 1;
    fill_arry(mat1, mat2, MATRIX_SIZE);
    test_hello();
    auto start = std::chrono::high_resolution_clock::now();

    matrix_mul((float*)mat1, (float*)mat2, (float*)result, MATRIX_SIZE);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "CPU matrix_mul " << duration.count() << " nanoseconds .\n";

}

void test_matrix_mul(){
    // initial matrix
    print_gpu_info();
    int row, col;
    row = MATRIX_SIZE;
    col = MATRIX_SIZE;
    float* mat1 = new float[MATRIX_SIZE*MATRIX_SIZE];
    float* mat2 = new float[MATRIX_SIZE*MATRIX_SIZE];
    float* result = new float[MATRIX_SIZE*MATRIX_SIZE];
    float* result_gpu = new float[MATRIX_SIZE*MATRIX_SIZE];
    fill_arry(mat1, mat2, MATRIX_SIZE);
    matrix_mul(mat1, mat2, result, MATRIX_SIZE);

    // cpy from host to device
    int size = row * col * sizeof(float);
    float *d_mat1, *d_mat2, *d_result;

    auto start = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_mat1, size);
    cudaMalloc(&d_mat2, size);
    cudaMalloc(&d_result, size);

    cudaMemcpy(d_mat1, mat1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, size, cudaMemcpyHostToDevice);

    // launch kernel
    launch_matmul(d_mat1, d_mat2, d_result, MATRIX_SIZE);

    // cpy result from device to host
    cudaMemcpy(result_gpu, d_result, size, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "GPU matrix_mul " << duration.count() << " nanoseconds .\n";
    // print result
    std::cout << "[GPU]:Result error:\n";
    for (int i = 0; i < row; ++i) {
        //for (int j = 0; j < col; ++j) {
            if(result_gpu[i*MATRIX_SIZE+0] != result[i*MATRIX_SIZE+0]){
                std::cout << "(" << i << "," << result[i*MATRIX_SIZE+0] << "," << result_gpu[i*MATRIX_SIZE+0] << ")" << "\t";
            }
        //}
    }
    std::cout << "\n";
    // clean
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);
}
