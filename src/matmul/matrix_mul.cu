#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
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

    for(int i = 0; i < N / TILE_DIM; i++){
        A_shared[threadIdx.y][threadIdx.x] = A[row*N+i*TILE_DIM+threadIdx.x];
        B_shared[threadIdx.y][threadIdx.x] = B[(i*TILE_DIM+threadIdx.y)*N+col];
        __syncthreads();

        for(int j = 0; j < TILE_DIM; j++){
            sum += A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x];
        }
    }

    // TODO: If N is not an integer multiple of TILE_DIM, the remaining part needs to be handled.
    C[row * N + col] = sum;
}

//Tiled and prefetch Matrix-Matrix Multiplication
__global__ void matmul_kernel_tiling_prefetch(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float A_shared[TILE_DIM][TILE_DIM];
    __shared__ float B_shared[TILE_DIM][TILE_DIM];
    float sum = 0.0f;
    float A_prefetch = A[row*N+threadIdx.x];
    float B_prefetch = B[threadIdx.y*N+col];
    for(int i = 0; i < N / TILE_DIM; i++){
        A_shared[threadIdx.y][threadIdx.x] = A_prefetch;
        B_shared[threadIdx.y][threadIdx.x] = B_prefetch;
        __syncthreads();

        if((i+1) < N/TILE_DIM){
            __pipeline_memcpy_async(&A_prefetch, &A[row*N+(i+1)*TILE_DIM+threadIdx.x], sizeof(float));
            __pipeline_memcpy_async(&B_prefetch, &B[((i+1)*TILE_DIM+threadIdx.y)*N+col], sizeof(float));
            __pipeline_commit();
            // A_prefetch = A[row*N+(i+1)*TILE_DIM+threadIdx.x];
            // B_prefetch = B[((i+1)*TILE_DIM+threadIdx.y)*N+col];
        }

        for(int j = 0; j < TILE_DIM; j++){
            sum += A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x];
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        //__syncthreads();
    }

    // TODO: If N is not an integer multiple of TILE_DIM, the remaining part needs to be handled.
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
        __pipeline_commit();
        __pipeline_wait_prior(0);
    }

    // TODO: If N is not an integer multiple of TILE_DIM, the remaining part needs to be handled.
    C[row * N + col] = sum;
}
//Todo: register, hardware(tile size, block size), tensor core, device-host data transfer, collaberative.
void launch_matmul(float* d_A, float* d_B, float* d_C, int N) {
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    //matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    //matmul_kernel_tiling<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    //matmul_kernel_tiling_prefetch<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    matmul_kernel_tiling_prefetch_bank<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
}

void fill_arry(float* A, float* B, int N){
    for(int i = 0; i < N; i++){
        for (int j = 0; j < N; j++) {
            A[i*N+j] = static_cast<float>(rand() % 5);
            B[i*N+j] = static_cast<float>(rand() % 5);
        }
    }
}
void test_matrix_mul_cpu(){
    int row, col;
    row = MATRIX_SIZE;
    col = MATRIX_SIZE;
    float mat1[row][col], mat2[row][col];
    float result[row][col];
    fill_arry((float*)mat1, (float*)mat2, MATRIX_SIZE);

    auto start = std::chrono::high_resolution_clock::now();

    matrix_mul((float*)mat1, (float*)mat2, (float*)result, MATRIX_SIZE);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "CPU matrix_mul " << duration.count() << " nanoseconds .\n";

}

void test_matrix_mul(){
    // initial matrix
    int row, col;
    row = MATRIX_SIZE;
    col = MATRIX_SIZE;
    float mat1[row][col], mat2[row][col];
    float result[row][col];
    fill_arry((float*)mat1, (float*)mat2, MATRIX_SIZE);
    matrix_mul((float*)mat1, (float*)mat2, (float*)result, MATRIX_SIZE);
    std::cout << "[CPU]:Result matrix C:\n";
    for (int i = 0; i < row; ++i) {
        //for (int j = 0; j < col; ++j) {
            std::cout << result[i][0] << "\t";
        //}
    }
    std::cout << "\n";

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
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "GPU matrix_mul " << duration.count() << " nanoseconds .\n";
    // print result
    std::cout << "[GPU]:Result matrix C:\n";
    for (int i = 0; i < row; ++i) {
        //for (int j = 0; j < col; ++j) {
            std::cout << result[i][0] << "\t";
        //}
    }
    std::cout << "\n";
    // clean
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);
}
