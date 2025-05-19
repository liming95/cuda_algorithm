#include <iostream>
#include <nvbench/nvbench.cuh>
#include <cuda/std/chrono>
#include <cuda_runtime.h>
#include <assert.h>
#include "matmul.cuh"

using namespace std;

__global__ void matmul_kernel_tiling(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_dim = blockDim.x;
    if (row >= N || col >= N || tile_dim != blockDim.y) return;

    extern __shared__ float shared_mem[]; 

    float* A_shared = shared_mem;
    float* B_shared = A_shared + tile_dim * tile_dim;

    float sum = 0.0f;

    for(int i = 0; i < N / tile_dim; i++){
        A_shared[threadIdx.y*tile_dim+threadIdx.x] = A[row*N+i*tile_dim+threadIdx.x];
        B_shared[threadIdx.y*tile_dim+threadIdx.x] = B[(i*tile_dim+threadIdx.y)*N+col];
        __syncthreads();

        for(int j = 0; j < tile_dim; j++){
            sum += A_shared[threadIdx.y*tile_dim+j] * B_shared[j*tile_dim+threadIdx.x];
        }
        __syncthreads();
    }

    // TODO: If N is not an integer multiple of tile_dim, the remaining part needs to be handled.
    C[row * N + col] = sum;
}

void matmul_tiling_bench(nvbench::state &state){
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
    int grid_dim = (matrix_size + block_dim - 1) / block_dim;
    dim3 blocksPerGrid(grid_dim, grid_dim);

    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(float);

    // Launch kernel
    state.exec(nvbench::exec_tag::timer, [blocksPerGrid,
        threadsPerBlock,
        shared_mem_size,
        d_mat1, d_mat2, d_result,
        matrix_dim
        ](nvbench::launch &launch, auto &timer) {
            if (!isValidLaunchConfig(blocksPerGrid, threadsPerBlock)) {
                std::cerr << "Invalid kernel configuration!" << std::endl;
            } else {
                timer.start();
                matmul_kernel_tiling<<<blocksPerGrid, threadsPerBlock, shared_mem_size,
                                                                launch.get_stream()>>>(d_mat1, 
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

NVBENCH_BENCH(matmul_tiling_bench)
  .add_int64_power_of_two_axis("MatrixDim", nvbench::range(5, 10, 1))
  .add_int64_power_of_two_axis("BlockDim", nvbench::range(3, 5, 1));
