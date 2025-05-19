#include <iostream>
#include <nvbench/nvbench.cuh>
#include <cuda/std/chrono>
#include <cuda_runtime.h>
#include <assert.h>
#include "matmul.cuh"

using namespace std;

#define REGISTER_MATMUL_TILING(TF)                                     \
void matmul_##TF##_tiling_bench(nvbench::state &state) {               \
    matmul_mul_tiling_bench<TF>(state);                                \
}                                                                       \
NVBENCH_BENCH(matmul_##TF##_tiling_bench)                               \
    .add_int64_power_of_two_axis("MatrixDim", nvbench::range(          \
        bench_para.min_matrix_dim, bench_para.max_matrix_dim, bench_para.strip_mat)) \
    .add_int64_power_of_two_axis("BlockDim", nvbench::range(           \
        bench_para.min_block_dim, bench_para.max_block_dim, bench_para.strip_block));

template<int TILE_FACTOR>
__global__ void matmul_kernel_mul_tiling(float* A, float* B, float* C, int N) {
    int row = TILE_FACTOR * blockIdx.y * blockDim.y + threadIdx.y;
    int col = TILE_FACTOR * blockIdx.x * blockDim.x + threadIdx.x;
    int tile_dim = blockDim.x * TILE_FACTOR;
    extern __shared__ float shared_mem[]; 

    float* A_shared = shared_mem;
    float* B_shared = A_shared + tile_dim * tile_dim;
    
    float sum[TILE_FACTOR * TILE_FACTOR];
    #pragma unroll
    for (int i = 0; i < TILE_FACTOR * TILE_FACTOR; i++) {
        sum[i] = 0.0f;
    }


    int stride = tile_dim / TILE_FACTOR;

    for(int i = 0; i < N / tile_dim; i++){
        int a_index, b_index, a_shar_index, b_shar_index;
        #pragma unroll
        for (int j = 0; j < TILE_FACTOR; j++){
            #pragma unroll
            for (int k = 0; k < TILE_FACTOR; k++){
                a_shar_index = (threadIdx.y + j * stride) * tile_dim + threadIdx.x + k * stride;
                b_shar_index = a_shar_index;
                a_index = (row + j * stride) * N + i * tile_dim + k * stride + threadIdx.x;
                b_index = (i * tile_dim + threadIdx.y + stride * j) * N + k * stride + col;

                A_shared[a_shar_index] = A[a_index];
                B_shared[b_shar_index] = B[b_index];
            }
        }

        __syncthreads();

        for(int j = 0; j < tile_dim; j++){
            int a_shar_index, b_shar_index, sum_index;
            #pragma unroll
            for (int z = 0; z < TILE_FACTOR; z++){
                #pragma unroll
                for (int k = 0; k < TILE_FACTOR; k++){
                    sum_index = z * TILE_FACTOR + k;
                    a_shar_index = (threadIdx.y + z * stride) * tile_dim + j;
                    b_shar_index = j * tile_dim + threadIdx.x + k * stride;

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

template<int TILE_FACTOR>
void matmul_mul_tiling_bench(nvbench::state &state){
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
    int tile_dim = block_dim * TILE_FACTOR;
    if (matrix_dim % tile_dim != 0) {
        std:: cerr << "Invalid matrix_dim for tile_dim" 
                   << "matrix dim:" << matrix_dim
                   << "tile_dim:" << tile_dim
                   << endl;
        return;
    }

    dim3 threadsPerBlock(block_dim, block_dim);
    int grid_dim = (matrix_dim + tile_dim - 1) / tile_dim;
    dim3 blocksPerGrid(grid_dim, grid_dim);


    size_t shared_mem_size = 2 * tile_dim * tile_dim * sizeof(float);

    // Launch kernel
    state.exec(nvbench::exec_tag::timer, [blocksPerGrid,
        threadsPerBlock,
        shared_mem_size,
        d_mat1, d_mat2, d_result,
        matrix_dim
        ](nvbench::launch &launch, auto &timer) {
            if (!isValidLaunchConfig(blocksPerGrid, threadsPerBlock)) {
                std::cerr << "[Error] Invalid kernel configuration.\n";
                return;
            }
            
            if (isSharedMemoryTooLarge(shared_mem_size)) {
                std::cerr << "[Error] Shared memory exceeds device limit.\n";
                return;
            }

            timer.start();
            matmul_kernel_mul_tiling<TILE_FACTOR><<<blocksPerGrid, threadsPerBlock, shared_mem_size,
                                                            launch.get_stream()>>>(d_mat1, 
                                                                                    d_mat2, 
                                                                                    d_result,
                                                                                    matrix_dim);                           
            timer.stop();
            CHECK_CUDA_ERROR("After kernel launch");
            CHECK_CUDA_SYNC("After device synchronize");
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

REGISTER_MATMUL_TILING(2)
REGISTER_MATMUL_TILING(4)
