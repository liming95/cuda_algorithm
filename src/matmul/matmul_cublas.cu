#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error: %d\n", status); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    const int M = 2, N = 3, K = 4; // Matrix dimensions: A[MxK], B[KxN], C[MxN]
    float alpha = 1.0f, beta = 0.0f;

    // Host matrices (row-major layout for C/C++)
    float h_A[M * K] = {
        1, 2, 3, 4,
        5, 6, 7, 8
    };
    float h_B[K * N] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    };
    float h_C[M * N] = {0}; // Output matrix

    // Device pointers
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

    // Copy input data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS handle creation
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    // cuBLAS uses column-major order, so we interpret input accordingly
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N, // No transpose
        N, M, K,                  // Dimensions in column-major
        &alpha,
        d_B, N,                   // B: (K x N), leading dimension = N
        d_A, K,                   // A: (M x K), leading dimension = K
        &beta,
        d_C, N                    // C: (M x N), leading dimension = N
    ));

    // Copy result matrix from device to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the result
    printf("Result matrix C:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

