#include <iostream>
#include <nvbench/nvbench.cuh>
#include <cuda/std/chrono>
#include <cuda_runtime.h>
#include "matmul.cuh"

using namespace std;

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


void matmul_gpu_bench(nvbench::state &state){
    const auto matrix_dim = static_cast<int>(state.get_int64("MatrixDim"));
    const auto block_dim = static_cast<unsigned int>(state.get_int64("BlockDim"));

    // initial matrix
    int matrix_size = matrix_dim * matrix_dim;
    float* mat1 = new float[matrix_size];
    float* mat2 = new float[matrix_size];
    float* result = new float[matrix_size];
    fill_arry(mat1, mat2, matrix_dim);


    // copy from host to device
    int size = matrix_size * sizeof(float);
    float *d_mat1, *d_mat2, *d_result;

    cudaMalloc(&d_mat1, size);
    cudaMalloc(&d_mat2, size);
    cudaMalloc(&d_result, size);
    cudaMemcpy(d_mat1, mat1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, size, cudaMemcpyHostToDevice);

    // set kernel config
    dim3 threadsPerBlock(block_dim, block_dim);
    int grid_dim = (matrix_dim + block_dim - 1) / block_dim;
    dim3 blocksPerGrid(grid_dim, grid_dim);

    // // Report throughput stats:
    // state.add_element_count(grid_dim);
    // state.add_global_memory_reads<nvbench::int32_t>(grid_dim);
    // state.add_global_memory_writes<nvbench::int32_t>(grid_dim);

    // Launch kernel
    state.exec(nvbench::exec_tag::timer, [blocksPerGrid,
        threadsPerBlock,
        d_mat1, d_mat2, d_result,
        matrix_dim
        ](nvbench::launch &launch, auto &timer) {
            if (!isValidLaunchConfig(blocksPerGrid, threadsPerBlock)) {
                std::cerr << "Invalid kernel configuration!" << std::endl;
            } else {   
                timer.start();
                matmul_kernel<<<blocksPerGrid, threadsPerBlock, 0, launch.get_stream()>>>(d_mat1, 
                                                                                        d_mat2, 
                                                                                        d_result,
                                                                                        matrix_dim);
                timer.stop();
                CHECK_CUDA_ERROR("After kernel launch");
                CHECK_CUDA_SYNC("After device synchronize");
            }
    });

    // copy result from device to host
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    
    // clean
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);
    delete [] mat1;
    delete [] mat2;
    delete [] result;
}

NVBENCH_BENCH(matmul_gpu_bench)
  .add_int64_power_of_two_axis("MatrixDim", nvbench::range(5, 10, 1))
  .add_int64_power_of_two_axis("BlockDim", nvbench::range(3, 5, 1));
