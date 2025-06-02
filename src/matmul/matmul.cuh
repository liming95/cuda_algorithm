#pragma once

#include <iostream>
#include <cuda_runtime.h>

struct {
    int max_matrix_dim;
    int min_matrix_dim;
    int max_block_dim;
    int min_block_dim;
    int strip_mat;
    int strip_block;
} bench_para = {
    9, // max_matrix_dim 2^10
    9,  // min_matrix_dim 2^5
    4,  // max_block_dim  2^5
    4,  // min_block_dim  2^3
    1,  // strip_mat
    1,  // strip_block
};



#define CHECK_CUDA_ERROR(msg)                                           \
    {                                                                   \
        cudaError_t err = cudaGetLastError();                           \
        if (err != cudaSuccess) {                                       \
            std::cerr << "[CUDA ERROR] " << msg << ": "                 \
                      << cudaGetErrorString(err) << " ("                \
                      << err << ")"                                     \
                      << " at " << __FILE__ << ":" << __LINE__         \
                      << std::endl;                                     \
        }                                                               \
    }

#define CHECK_CUDA_SYNC(msg)                                            \
    {                                                                   \
        cudaError_t err = cudaDeviceSynchronize();                      \
        if (err != cudaSuccess) {                                       \
            std::cerr << "[CUDA SYNC ERROR] " << msg << ": "            \
                      << cudaGetErrorString(err) << " ("                \
                      << err << ")"                                     \
                      << " at " << __FILE__ << ":" << __LINE__         \
                      << std::endl;                                     \
        }                                                               \
    }

// initialize the matrix
void fill_arry(float* A, float* B, int N){
    for(int i = 0; i < N; i++){
        for (int j = 0; j < N; j++) {
            A[i*N+j] = static_cast<float>(rand() % 5) + 1;
            B[i*N+j] = static_cast<float>(rand() % 5) + 1;

        }
    }
}


#include <iostream>
#include <cuda_runtime.h>

bool isValidLaunchConfig(dim3 gridDim, dim3 blockDim) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // get properties on device 0

    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int maxBlockDimX = prop.maxThreadsDim[0];
    int maxBlockDimY = prop.maxThreadsDim[1];
    int maxBlockDimZ = prop.maxThreadsDim[2];
    int maxGridDimX = prop.maxGridSize[0];
    int maxGridDimY = prop.maxGridSize[1];
    int maxGridDimZ = prop.maxGridSize[2];

    // threads in blocks
    unsigned int totalThreads = blockDim.x * blockDim.y * blockDim.z;
    if (static_cast<int>(totalThreads) > maxThreadsPerBlock) {
        std::cerr << "Error: total threads per block (" << totalThreads
                  << ") exceeds max (" << maxThreadsPerBlock << ")\n";
        return false;
    }

    // block dim
    if (static_cast<int>(blockDim.x) > maxBlockDimX ||
        static_cast<int>(blockDim.y) > maxBlockDimY ||
        static_cast<int>(blockDim.z) > maxBlockDimZ) {
        std::cerr << "Error: blockDim exceeds max per-dimension limits: "
                  << "(" << blockDim.x << "," << blockDim.y << "," << blockDim.z << ") vs max ("
                  << maxBlockDimX << "," << maxBlockDimY << "," << maxBlockDimZ << ")\n";
        return false;
    }

    // grid dim
    if (static_cast<int>(gridDim.x) > maxGridDimX ||
        static_cast<int>(gridDim.y) > maxGridDimY ||
        static_cast<int>(gridDim.z) > maxGridDimZ) {
        std::cerr << "Error: gridDim exceeds max per-dimension limits: "
                  << "(" << gridDim.x << "," << gridDim.y << "," << gridDim.z << ") vs max ("
                  << maxGridDimX << "," << maxGridDimY << "," << maxGridDimZ << ")\n";
        return false;
    }

    return true;
}

bool isSharedMemoryTooLarge(size_t shared_mem_bytes) {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    size_t max_shared_mem = prop.sharedMemPerBlock;

    if (shared_mem_bytes > max_shared_mem) {
        std::cerr << "[Error] Shared memory required (" << shared_mem_bytes
                  << " bytes) exceeds device limit (" << max_shared_mem << " bytes).\n";
        return true;
    } else {
        return false;
    }
}
