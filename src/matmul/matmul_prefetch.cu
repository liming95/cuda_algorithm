#include <iostream>
#include <nvbench/nvbench.cuh>
#include <cuda/std/chrono>
#include <cuda_runtime.h>
#include <assert.h>
#include <cuda_pipeline.h>
#include "matmul.cuh"

using namespace std;
#define  NUM_BUFFERS 2

__global__ void matmul_kernel_tiling_prefetch(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_dim = blockDim.x;
    if (row >= N || col >= N || tile_dim != blockDim.y) return;

    extern __shared__ float shared_mem[]; 

    int tile_size = tile_dim * tile_dim;
    float* A_shared = shared_mem;
    float* B_shared = A_shared + NUM_BUFFERS * tile_size;

    float sum = 0.0f;
    int cur_buf = 0;

    int index_in_line = threadIdx.y * tile_dim + threadIdx.x;
    int a_index = row * N + threadIdx.x;
    int b_index = threadIdx.y * N + col;

    A_shared[index_in_line] = A[a_index];
    B_shared[index_in_line] = B[b_index];
    __syncthreads();

    for (int i = 0; i < N / tile_dim; i++) {
        int next = cur_buf ^ 1;
        int next_index_in_line = next * tile_size + index_in_line;
        if ((i + 1) < N / tile_dim) {
            a_index = row * N + (i + 1) * tile_dim + threadIdx.x;
            b_index = ((i + 1) * tile_dim + threadIdx.y) * N + col;

            __pipeline_memcpy_async(&A_shared[next_index_in_line], &A[a_index], sizeof(float));
            __pipeline_memcpy_async(&B_shared[next_index_in_line], &B[b_index], sizeof(float));
            __pipeline_commit();
        }

        for (int j = 0; j < tile_dim; ++j) {
            sum += A_shared[cur_buf*tile_size+threadIdx.y*tile_dim+j] * 
                    B_shared[cur_buf*tile_size+j*tile_dim+threadIdx.x];
        }

        __pipeline_wait_prior(0);
        __syncthreads();
        cur_buf ^= 1;
    }
   C[row * N + col] = sum;
}

void matmul_prefetch_bench(nvbench::state &state){
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

    size_t shared_mem_size = 2 * NUM_BUFFERS * block_dim * block_dim * sizeof(float);

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
                matmul_kernel_tiling_prefetch<<<blocksPerGrid, threadsPerBlock, shared_mem_size,
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

NVBENCH_BENCH(matmul_prefetch_bench)
  .add_int64_power_of_two_axis("MatrixDim", nvbench::range(bench_para.min_matrix_dim, bench_para.max_matrix_dim, bench_para.strip_mat))
  .add_int64_power_of_two_axis("BlockDim", nvbench::range(bench_para.min_block_dim, bench_para.max_block_dim, bench_para.strip_block));

