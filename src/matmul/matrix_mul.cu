#include <iostream>
#include <cstdlib>
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

void launch_matmul(float* d_A, float* d_B, float* d_C, int N) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
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
    matrix_mul((float*)mat1, (float*)mat2, (float*)result, MATRIX_SIZE);

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
        for (int j = 0; j < col; ++j) {
            std::cout << result[i][j] << "\t";
        }
        std::cout << "\n";
    }

    // cpy from host to device
    int size = row * col * sizeof(float);
    float *d_mat1, *d_mat2, *d_result;

    cudaMalloc(&d_mat1, size);
    cudaMalloc(&d_mat2, size);
    cudaMalloc(&d_result, size);

    cudaMemcpy(d_mat1, mat1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, size, cudaMemcpyHostToDevice);

    // launch kernel
    launch_matmul(d_mat1, d_mat2, d_result, MATRIX_SIZE);

    // cpy result from device to host
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

    // print result
    std::cout << "[GPU]:Result matrix C:\n";
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            std::cout << result[i][j] << "\t";
        }
        std::cout << "\n";
    }

    // clean
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);
}
