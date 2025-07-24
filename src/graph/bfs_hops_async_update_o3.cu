#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>

#define DEBUG_LEVEL 0
#include "bfs_hops.cuh"

inline void calculate_kernel_config(int thread_num, int& block_size, int& grid_size){
    //int threadsPerBlock_up = ((thread_num + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    //block_size = thread_num > BLOCK_MAX_SIZE ? BLOCK_MAX_SIZE : threadsPerBlock_up;
    block_size = BLOCK_MAX_SIZE;
    grid_size = (thread_num + BLOCK_MAX_SIZE - 1) / BLOCK_MAX_SIZE;
}

__device__ int update_frontiers_num;

__global__ void initial_output_bitmap_o3(int* output_num, int update_num){
    int glb_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(glb_tid == 0){
        update_frontiers_num += update_num;
        *output_num = 0;
    }

}

#define PARA_BLK_OUTPUT_SIZE 8*BLOCK_MAX_SIZE
#define BITMAP_TIME 5
#define PARA_BLK_BITMAP_SIZE BLOCK_MAX_SIZE*BITMAP_TIME
__global__ void get_output_frontiers_o3(int* g_offset, int* g_edges, int node_num, int edge_num,
                                    int* input_frontiers, int input_num,
                                    int* output_frontiers, int* output_num,
                                    int* update_frontiers,
                                    int* status_bitmap, int bitmap_len ){
    int glb_tid = blockIdx.x * blockDim.x + threadIdx.x;
    CUDA_DEBUG(glb_tid, "Get Output Frontiers O1...\n");
    int blk_tid = threadIdx.x;
    int block_size = blockDim.x;
    //int grid_size = gridDim.x;
    //int bid = blockIdx.x;

    //typedef cub::BlockReduce<int, BLOCK_MAX_SIZE> BlockReduce;
    typedef cub::BlockScan<int, BLOCK_MAX_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    //__shared__ typename BlockReduce::TempStorage temp_storage2;
    __shared__ int blk_output_bitmap[PARA_BLK_BITMAP_SIZE];
    __shared__ int blk_input_frontiers[BLOCK_MAX_SIZE];

    // 1. load input vertex & initial
    if(glb_tid < input_num) {
        blk_input_frontiers[blk_tid] = input_frontiers[glb_tid];
    }
    for(int i = 0; i < BITMAP_TIME; i++)
        blk_output_bitmap[i*BLOCK_MAX_SIZE+blk_tid] = 0;

    // 2.travel and update blk_output_bitmap
    int bitmap;
    int neighbors_num;
    int vertex, start, end, neighbor;
    int total_ngbs;
    int index_in_bitmap, offset_in_bitmap;

    __shared__ int blk_degrees[BLOCK_MAX_SIZE];
    __shared__ int blk_start_offset[BLOCK_MAX_SIZE];

    if(glb_tid < input_num){
        vertex = blk_input_frontiers[blk_tid];
        start = g_offset[vertex];
        blk_start_offset[blk_tid] = start;
        end = g_offset[vertex+1];
        neighbors_num = end - start;
        DEBUG_PRINT("(2.1) blk tid:(%d, %d), vertex: %d, neighbor num: %d\n", blockIdx.x, blk_tid, vertex, neighbors_num);
    } else {
        neighbors_num = 0;
    }

    BlockScan(temp_storage).ExclusiveSum(neighbors_num, neighbors_num, total_ngbs);
    blk_degrees[blk_tid] = neighbors_num;
    __syncthreads();
    CUDA_DEBUG_BLK(blk_tid, "(2.1) total neighbors num: %d\n", total_ngbs);

    for(int i = blk_tid; i < total_ngbs; i += block_size){
        auto it = thrust::upper_bound(thrust::seq, blk_degrees, blk_degrees+BLOCK_MAX_SIZE, i);
        int idx = thrust::distance(blk_degrees, it) - 1;

        vertex = blk_input_frontiers[idx];
        start = blk_start_offset[idx];
        int offset_in_ngb_per_vertex = i - blk_degrees[idx];
        neighbor = g_edges[start+offset_in_ngb_per_vertex];
        DEBUG_PRINT("(2.1) vertex: %d, neighbor: %d\n", vertex, neighbor);

        // mark output bitmap
        GET_BIT_INDEX_OFFSET(neighbor, index_in_bitmap, offset_in_bitmap);
        int bitmask = 1 << offset_in_bitmap;

        bool cached = index_in_bitmap < PARA_BLK_BITMAP_SIZE;

        bitmap = cached ? blk_output_bitmap[index_in_bitmap] :
                    atomicOr(&status_bitmap[index_in_bitmap], bitmask);

        bool filled = bitmap & bitmask;
        // if(filled) continue;

        int s_bitmap = cached && !filled ? atomicOr(&status_bitmap[index_in_bitmap], bitmask) : bitmap;
        filled = s_bitmap & bitmask;
        // if(filled) continue;

        // bitmap = s_bitmap & bitmask;
        bitmap = cached && (s_bitmap != bitmap) ? atomicOr(&blk_output_bitmap[index_in_bitmap], s_bitmap & bitmask) : bitmap;

        //filled = filled ? filled : bitmap & bitmask;

        if(!filled) {
            int index = atomicAdd(output_num, 1);
            output_frontiers[index] = neighbor;
            update_frontiers[update_frontiers_num+index] = neighbor;
            DEBUG_PRINT("(2.2) pre_output_num: %d, vertex: %d\n", index, neighbor);
        }





        // int thread_mask = __match_any_sync(__activemask(), index_in_bitmap);
        // int first_tid = __ffs(thread_mask) - 1;
        // int warp_or = __reduce_or_sync(thread_mask, bitmask);

        // if(blk_tid % 32 == first_tid){
        //     if(index_in_bitmap < PARA_BLK_BITMAP_SIZE) {
        //         atomicOr(&blk_output_bitmap[index_in_bitmap], warp_or);
        //         warp_or = 0;
        //     }
        //     else {
        //         bitmap = atomicOr(&status_bitmap[index_in_bitmap], warp_or);
        //         bitmap ^= warp_or;
        //         warp_or &= bitmap;
        //     }
        // }


        // warp_or = __shfl_sync(thread_mask, warp_or, first_tid);
        // thread_mask = __match_any_sync(__activemask(), neighbor);
        // first_tid = __ffs(thread_mask) - 1;
        // // eliminate repeated element
        // if((blk_tid % 32 == first_tid) && (warp_or & bitmask)) {
        //     // Todo: write by batch
        //     int index = atomicAdd(output_num, 1);
        //     output_frontiers[index] = neighbor;
        //     update_frontiers[update_frontiers_num+index] = neighbor;
        //     DEBUG_PRINT("(2.2) pre_output_num: %d, vertex: %d\n", index, neighbor);
        // }
    }
    // __syncthreads();

    // 3. aggregate the blk output bitmap to output bitmap
    // int it_time = BLOCK_MAX_SIZE / BLOCK_MAX_SIZE;

    // int inserted_num;
    // int inserted_offset; //prefix sum
    // int pre_bitmap;
    // int block_sum;
    // int blk_output_num = 0;
    // int bitmap_idx;
    // //__shared__ int thread_output_num[BLOCK_MAX_SIZE];
    // __shared__ int blk_output_frontiers[PARA_BLK_OUTPUT_SIZE];

    // for(int i = 0; i < BITMAP_TIME; i ++){
    //     bitmap_idx = i * BLOCK_MAX_SIZE + blk_tid;
    //     bitmap = blk_output_bitmap[bitmap_idx];
    //     pre_bitmap = bitmap == 0 ? 0 : atomicOr(&status_bitmap[bitmap_idx], bitmap);
    //     DEBUG_PRINT("(3.1) blk(%d, %d), pre_bitmap: %x, bitmap: %x\n", blockIdx.x, blk_tid, pre_bitmap, bitmap);

    //     pre_bitmap ^= bitmap;
    //     bitmap &= pre_bitmap;
    //     //blk_output_bitmap[i*BLOCK_MAX_SIZE+blk_tid] = bitmap;
    //     inserted_num = __popc(bitmap);

    //     BlockScan(temp_storage).ExclusiveSum(inserted_num, inserted_offset, block_sum);

    //     // bitmap-> frontiers
    //     //Todo: the imbalance between bitmap
    //     int pos;
    //     for(int j = 0; j < inserted_num; j++) {
    //         int output_offset = blk_output_num + inserted_offset + j;
    //         pos = __ffs(bitmap) - 1;
    //         pos = bitmap_idx * 32 + pos;

    //         if(output_offset < PARA_BLK_OUTPUT_SIZE) blk_output_frontiers[output_offset] = pos;
    //         else {
    //             int index = atomicAdd(output_num, 1);
    //             output_frontiers[index] = pos;
    //             update_frontiers[update_frontiers_num+index] = pos;
    //             DEBUG_PRINT("(3.2) pre_output_num: %d, vertex: %d\n", index, neighbor);
    //         }
    //         //assert(bitmap != 0);
    //         bitmap &= bitmap - 1;
    //     }

    //     blk_output_num += block_sum;
    //     blk_output_num = blk_output_num > PARA_BLK_OUTPUT_SIZE ? PARA_BLK_OUTPUT_SIZE : blk_output_num;
    //     DEBUG_PRINT("(3.3) blk(%d, %d) pre_bitmap: %x, bitmap: %x, inserted_num: %d, block_sum: %d\n", blockIdx.x, blk_tid, pre_bitmap, bitmap, inserted_num, block_sum);


    //     // load balance when access bitmap
    //     // thread_output_num[blk_tid] = inserted_num;
    //     // blk_degrees[blk_tid] = inserted_num;
    //     // __syncthreads();

    //     // for(int j = blk_tid; j < block_sum; j += block_size) {
    //     //     auto it = thrust::upper_bound(thrust::seq, blk_degrees, blk_degrees+BLOCK_MAX_SIZE, j);
    //     //     int idx = thrust::distance(blk_degrees, it) - 1;
    //     //     int bitmap_index = i * BLOCK_MAX_SIZE + idx;

    //     //     bitmap = blk_output_bitmap[bitmap_index];
    //     //     int offset_in_bitmap = j - blk_degrees[idx];
    //     //     int bitmap_popc = thread_output_num[idx];

    //     //     int bit_len_per_part = 32 / bitmap_popc;
    //     //     int bit_rmd = 32 % bitmap_popc;
    //     //     int bit_offset = bit_len_per_part * offset_in_bitmap;
    //     //     bit_offset += offset_in_bitmap < bit_rmd ? offset_in_bitmap : bit_rmd;
    //     //     bit_len_per_part += offset_in_bitmap < bit_rmd ? 1 : 0;
    //     //     bitmap = (bitmap >> bit_offset) << bit_offset;
    //     //     int rmd_len = 32 - bit_offset - bit_len_per_part;
    //     //     bitmap = ((unsigned int)(bitmap << rmd_len)) >> rmd_len;

    //     //     int pos;
    //     //     while(bitmap) {
    //     //        pos = __ffs(bitmap) - 1;
    //     //        pos = bitmap_index * 32 + pos;
    //     //        int index = atomicAdd(output_num, 1);
    //     //        output_frontiers[index] = pos;
    //     //        bitmap &= bitmap -1;
    //     //     }
    //     // }

    // }

    // // 4.add output vertex to update frontiers & output frontiers.
    // __shared__ int output_offset;
    // __shared__ int blk_update_offset;
    // if(blk_tid == 0) {
    //     output_offset = atomicAdd(output_num, blk_output_num);
    //     CUDA_DEBUG_BLK(blk_tid, "(4.1) pre_output_num: %d, blk_output_num: %d, output num: %d\n", output_offset, blk_output_num, *output_num);
    //     blk_update_offset = update_frontiers_num + output_offset;
    //     CUDA_DEBUG_BLK(blk_tid, "(4.1) pre_update_num: %d, offset: %d\n", update_frontiers_num, output_offset);
    // }

    // __syncthreads();

    // for(int i = blk_tid; i < blk_output_num; i += block_size){
    //     update_frontiers[blk_update_offset+i] = blk_output_frontiers[i];
    //     output_frontiers[output_offset+i] = blk_output_frontiers[i];
    //     DEBUG_PRINT("(4.2) output offset: %d, vertex: %d\n", output_offset, blk_output_frontiers[i]);
    // }
}

__global__ void update_node_status_o3(int* input_frontiers, int offset, int input_num, int* hops, int hop){
    int glb_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(glb_tid < input_num){
        int node = input_frontiers[offset+glb_tid];
        DEBUG_PRINT("offset: %d, node: %d, pre_hop: %d, hop: %d\n", offset+glb_tid, node, hops[node], hop);
        hops[node] = hop;
    }
}

void bfs_hops_async_o3(std::vector<int> offset, std::vector<int> edges, int node_num, int edge_num,
                   int source,
                   std::vector<int> &hops){
    int* d_offset, *d_edges;
    int offset_size = offset.size() * sizeof(int);
    int edge_size = edge_num * sizeof(int);

    int* d_input_frontiers; //int* d_nf_num;
    int* d_output_frontiers;
    int* d_status_bitmap;
    int word_bit_len = sizeof(int) * BYTE_SIZE;
    int bitmap_len = (node_num + word_bit_len - 1) / word_bit_len;
    int bitmap_size = bitmap_len * sizeof(int);
    int* bitmap = new int[bitmap_len];

    int* d_hops;
    int hops_size = node_num * sizeof(int);

    int* d_update_frontiers;
    int nf_size = node_num * sizeof(int);

    int* d_output_num; int output_num;
    int nf_num_size = sizeof(int);

    cudaMalloc(&d_offset, offset_size);
    cudaMalloc(&d_edges, edge_size);
    cudaMalloc(&d_input_frontiers, nf_size);
    cudaMalloc(&d_output_frontiers, nf_size);
    cudaMalloc(&d_status_bitmap, bitmap_size);
    cudaMalloc(&d_output_num, nf_num_size);
    cudaMalloc(&d_hops, hops_size);
    cudaMalloc(&d_update_frontiers, nf_size);
    // initial the input value
    cudaMemcpy(d_offset, offset.data(), offset_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, edges.data(), edge_size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_input_frontiers, &source, sizeof(int), cudaMemcpyHostToDevice);

    int index_in_bitmap, offset_in_bitmap;
    GET_BIT_INDEX_OFFSET(source, index_in_bitmap, offset_in_bitmap);
    int bitmask = 1 << offset_in_bitmap;
    for(int i = 0; i < bitmap_len; i++){
        bitmap[i] = 0;
    }
    bitmap[index_in_bitmap] = bitmask;
    DEBUG_PRINT("initial input bitmap: index(%d), value(%d)\n", index_in_bitmap, bitmask);
    cudaMemcpy(d_status_bitmap, bitmap, bitmap_size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_hops, hops.data(), hops_size, cudaMemcpyHostToDevice);

    int initial_value = 0; int update_offset = 1;
    cudaMemcpyToSymbol(update_frontiers_num, &initial_value, sizeof(int));

    cudaStream_t stream_traversal, stream_update;
    cudaStreamCreate(&stream_traversal);
    cudaStreamCreate(&stream_update);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //int sum = 0, level = 0;
    int threadsPerBlock, blocksPerGrid;
    int cur_hop = 0;
    int input_num = 1;
    calculate_kernel_config(input_num, threadsPerBlock, blocksPerGrid);
    cudaEventRecord(start, 0);
    // int threadsPerBlock_up = ((bitmap_len + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    // int threadsPerBlock_inital = bitmap_len > BLOCK_MAX_SIZE ? BLOCK_MAX_SIZE : threadsPerBlock_up;
    // int blocksPerGrid_inital = (bitmap_len + BLOCK_MAX_SIZE - 1) / BLOCK_MAX_SIZE;

    while(input_num){
        //int smem_size = (2 * bitmap_len + ((bitmap_len + blocksPerGrid - 1) / blocksPerGrid) * 32) * sizeof(int);
        //assert(smem_size < 48 * 1024);

        initial_output_bitmap_o3<<<1, 32>>>(d_output_num, input_num);
        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     printf("Kernel launch 1 error: %s\n", cudaGetErrorString(err));
        // }
        calculate_kernel_config(input_num, threadsPerBlock, blocksPerGrid);
        get_output_frontiers_o3<<<blocksPerGrid, threadsPerBlock>>>(d_offset, d_edges, node_num, edge_num,
                                                                                d_input_frontiers, input_num,
                                                                                d_output_frontiers, d_output_num,
                                                                                d_update_frontiers,
                                                                                d_status_bitmap, bitmap_len
                                                                                );
        // err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     printf("Kernel launch 2 error: %s\n", cudaGetErrorString(err));
        // }

        cudaMemcpy(&output_num, d_output_num, nf_num_size, cudaMemcpyDeviceToHost);
        // err = cudaDeviceSynchronize();
        // if (err != cudaSuccess) {
        //     printf("CUDA error: %s\n", cudaGetErrorString(err));
        //     //return;
        // }
        cur_hop++;
        calculate_kernel_config(output_num, threadsPerBlock, blocksPerGrid);
        update_node_status_o3<<<blocksPerGrid, threadsPerBlock, 0, stream_update>>>(d_update_frontiers, update_offset, output_num,
                                                                   d_hops, cur_hop);
        update_offset += output_num;

        input_num = output_num;
        int* tmp = d_input_frontiers;
        d_input_frontiers = d_output_frontiers;
        d_output_frontiers = tmp;
        DEBUG_PRINT("level: %d, output_num: %d\n", cur_hop, input_num);
        //printf("update_offset: %d\n", update_offset);
    }
    //printf("sum: %d\n", sum);
    cudaMemcpy(hops.data(), d_hops, hops_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU: bfs_hops_async_o3. Elapsed time :" << milliseconds << " (ms)\n";

    cudaFree(d_offset);
    cudaFree(d_edges);
    cudaFree(d_hops);
    cudaFree(d_input_frontiers);
    cudaFree(d_output_frontiers);
    cudaFree(d_status_bitmap);
    cudaFree(d_output_num);
    cudaFree(d_update_frontiers);

    cudaStreamDestroy(stream_traversal);
    cudaStreamDestroy(stream_update);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

std::vector<int> test_bfs_hops_async_o3(std::vector<int> offset, std::vector<int> endnodes, int source){
    int node_num = offset.size() - 1;
    int edge_num = endnodes.size();

    std::vector<int> hops(node_num, INVAILD);
    hops[source] = 0;
    // std::cout << "GPU:" << std::endl;

    bfs_hops_async_o3(offset, endnodes, node_num, edge_num, source, hops);

    //print_hops(source, hops);
    return hops;
}
