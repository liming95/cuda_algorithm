#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>
#include "bfs_hops.cuh"

inline void calculate_kernel_config(int thread_num, int& block_size, int& grid_size){
    //int threadsPerBlock_up = ((thread_num + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    //block_size = thread_num > BLOCK_MAX_SIZE ? BLOCK_MAX_SIZE : threadsPerBlock_up;
    block_size = BLOCK_MAX_SIZE;
    grid_size = (thread_num + BLOCK_MAX_SIZE - 1) / BLOCK_MAX_SIZE;
}

__device__ int update_frontiers_num;

__global__ void initial_output_bitmap(int* output_bitmap, int* output_num, int bitmap_len){
    int glb_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_size = blockDim.x;
    int grid_size = gridDim.x;
    int threadsPerGrid = block_size * grid_size;

    for (int i = glb_tid; i < bitmap_len; i += threadsPerGrid) {
        output_bitmap[i] = 0;
    }
    if(glb_tid == 0) &output_num = 0;
}

__global__ void get_output_frontiers_o1(int* g_offset, int* g_edges, int node_num, int edge_num,
                                    int* input_bitmap, int nf_num,
                                    int* output_bitmap, int* output_num,
                                    int* status_bitmap, int bitmap_len,
                                    int* update_frontiers){
    int glb_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blk_tid = threadIdx.x;
    int block_size = blockDim.x;
    int grid_size = gridDim.x;
    int bid = blockIdx.x;

    int node, neighbor;
    int edge_start, edge_end;
    int index;
    int index_in_bitmap, offset_in_bitmap;

    typedef cub::BlockReduce<int, 256> BlockReduce;
    typedef cub::BlockScan<int, BLOCK_MAX_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ int blk_output_num;

    //Todo: sparse store
    extern __shared__ int dynamic_smem[];
    int* blk_status_bitmap = dynamic_smem;
    int* blk_output_bitmap = dynamic_smem + bitmap_len;
    int* blk_input_frontiers = dynamic_smem + 2 * bitmap_len;


    // 1. load bitmap into shared memory (Todo:async)
    for (int i = blk_tid; i < bitmap_len; i += block_size){
        blk_status_bitmap[i] = status_bitmap[i];
        blk_output_bitmap[i] = 0;
    }
    if(blk_tid == 0) blk_output_num = 0;
    __syncthreads();

    // 2. get input frontiers
    int bitmap;
    int v_num[1];
    int v_offset[1];
    int total_v_num_per_it;
    int total_input_num_in_blk = 0;

    int input_bitmap_remaind = bitmap_len % grid_size;
    int input_bitmap_num = bitmap_len / grid_size;
    int input_bitmap_offset = bid * input_bitmap_num;
    // assign bitmap
    if (bid < input_bitmap_remaind) {
        input_bitmap_num += 1;
        input_bitmap_offset += bid;
    } else {
        input_bitmap_offset += input_bitmap_remaind;
    }

    int it = input_bitmap_num / block_size;
    int blk_input_bitmap_remaind = input_bitmap_num % block_size

    // the block-length bitmap
    for(int i = 0; i < it; i++){
        bitmap = input_bitmap[input_bitmap_offset+blk_id];
        v_num[0] = __popc(bitmap);

        BlockScan(temp_storage).ExclusiveSum(v_num, v_offset, total_v_num_per_it);
        __syncthreads();

        //Todo: binary search in bitmap
        int pos;
        for(int j = 0; j < v_num[0]; j++) {
            pos = __ffs(bitmap) - 1;
            pos = i * 32 + pos;
            blk_input_frontiers[total_input_num_in_blk+v_offset+j] = pos;
            bitmap &= bitmap - 1;
        }

        total_input_num_in_blk += total_v_num_per_it;
        input_bitmap_offset += block_size;
    }
    // the remainding bitmap
    if(blk_tid < blk_input_bitmap_remaind) {
        bitmap = input_bitmap[input_bitmap_offset+blk_tid];
        v_num[0] = __popc(bitmap);
    } else {
        v_num[0] = 0;
    }

    BlockScan(temp_storage).ExclusiveSum(v_num, v_offset, total_v_num_per_it);
    __syncthreads();

    //Todo: binary search for load balance
    int pos;
    for(int j = 0; j < v_num[0]; j++) {
        pos = __ffs(bitmap) - 1;
        pos = i * 32 + pos;
        blk_input_frontiers[total_input_num_in_blk+v_offset+j] = pos;
        bitmap &= bitmap - 1;
    }

    total_input_num_in_blk += total_v_num_per_it;

    //3.add vertex to update frontiers
    __shared__ int blk_update_offset;
    if (blk_tid == 0) {
        blk_update_offset = atomicAdd(update_frontiers_num, total_input_num_in_blk);
    }
    __syncthreads();

    for(int i = blk_tid; i < total_input_num_in_blk; i += block_size){
        update_frontiers[blk_update_offset+i] = blk_input_frontiers[i];
    }

    //4.travel and update blk_output_bitmap
    int it_time = total_input_num_in_blk / block_size;
    int blk_input_frontiers_remaind = total_input_num_in_blk % block_size;
    int neighbors_num;
    int vertex, start, end;
    int total_ngbs;

    __shared__ int blk_degrees[BLOCK_MAX_SIZE];
    __shared__ int blk_start_offset[BLOCK_MAX_SIZE];

    for(int i = 0; i < it_time ; i++) {
        vertex = blk_input_frontiers[block_size*i+blk_tid];
        start = g_offset[vertex];
        blk_start_offset[blk_tid] = start;
        end = g_offset[vertex+1];
        neighbors_num = start - end;

        BlockScan(temp_storage).ExclusiveSum(neighbors_num, neighbors_num, total_ngbs);
        blk_degrees[blk_tid] = neighbors_num;
        __syncthreads();

        for(int j = blk_tid; j < total_ngbs; j += block_size){
            auto it = thrust::upper_bound(thrust::seq, blk_degrees, blk_degrees+block_size, j);
            int idx = thrust::distance(blk_degrees, it) - 1;

            vertex = blk_input_frontiers[block_size*i+idx];
            start = blk_start_offset[idx];
            int offset_in_ngb_per_vertex = j - blk_degrees[idx];
            neighbor = g_edges[start+offset_in_ngb_per_vertex];

            // mark output bitmap
            GET_BIT_INDEX_OFFSET(neighbor, index_in_bitmap, offset_in_bitmap);
            int bitmask = 1 << offset_in_bitmap;
            int thread_mask = __match_any_sync(__activemask(), index_in_bitmap);
            int first_tid = __ffs(thread_mask) - 1;
            int warp_or = __reduce_or_sync(thread_mask, bitmask);
            if(blk_tid % 32 == first_tid){
                bitmap = blk_status_bitmap[index_in_bitmap];
                bitmap ^= warp_or;
                warp_or &= bitmap;
                atomicOr(&blk_output_bitmap[index_in_bitmap], warp_or);
            }
        }
    }

    // the remainding vertex
    if(blk_tid < blk_input_frontiers_remaind){
        vertex = input_frontiers[it_time*block_size+blk_tid];
        start = g_offset[vertex];
        blk_start_offset[blk_tid] = start;
        end = g_offset[vertex+1];
        neighbors_num = start - end;
    } else {
        neighbors_num = 0;
    }

    BlockScan(temp_storage).ExclusiveSum(neighbors_num, neighbors_num, total_ngbs);
    blk_degrees[blk_tid] = neighbors_num;
    __syncthreads();

    for(int i = blk_tid; i < total_ngbs; i += block_size){
        auto it = thrust::upper_bound(thrust::seq, blk_degrees, blk_degrees+blk_input_frontiers_remaind, i);
        int idx = thrust::distance(blk_degrees, it) - 1;

        vertex = blk_input_frontiers[block_size*it_time+idx];
        start = blk_start_offset[idx];
        int offset_in_ngb_per_vertex = i - blk_degrees[idx];
        neighbor = g_edges[start+offset_in_ngb_per_vertex];

        // mark output bitmap
        GET_BIT_INDEX_OFFSET(neighbor, index_in_bitmap, offset_in_bitmap);
        int bitmask = 1 << offset_in_bitmap;
        int thread_mask = __match_any_sync(__activemask(), index_in_bitmap);
        int first_tid = __ffs(thread_mask) - 1;
        int warp_or = __reduce_or_sync(thread_mask, bitmask);
        if(blk_tid % 32 == first_tid){
            bitmap = blk_status_bitmap[index_in_bitmap];
            bitmap ^= warp_or;
            warp_or &= bitmap;
            atomicOr(&blk_output_bitmap[index_in_bitmap], warp_or);
        }
    }
    __syncthreads();

    //5. aggregate the blk output bitmap to output bitmap
    for(int i = blk_tid; i < bitmap_len; i += block_size){
        bitmap = blk_output_bitmap[i];
        int pre_bitmap = bitmap == 0 ? 0 : atomicOr(&status_bitmap[i], bitmap);
        int pre_bitmap = bitmap == 0 ? 0 : atomicOr(&output_bitmap[i], bitmap);

        pre_bitmap ^= bitmap;
        bitmap &= pre_bitmap;
        int inserted_num = __popc(bitmap);
        int block_sum = BlockReduce(temp_storage).Sum(inserted_num);

        if(blk_id == 0) {
            blk_output_num += block_sum;
        }
    }

    atomicAdd(&output_num, blk_output_num);
}

__global__ void update_node_status_o1(int* input_bitmap, int offset, int nf_num, int* hops, int hop){
    int glb_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(glb_tid < nf_num){
        int node = input_bitmap[offset+glb_tid];
        //printf("node: %d, hop: %d\n", node, hops[node]);
        hops[node] = hop + 1;
    }
}

void bfs_hops_async_o1(std::vector<int> offset, std::vector<int> edges, int node_num, int edge_num,
                   int source,
                   std::vector<int> &hops){
    int* d_offset, *d_edges;
    int offset_size = offset.size() * sizeof(int);
    int edge_size = edge_num * sizeof(int);

    int* d_input_bitmap; //int* d_nf_num;
    int* d_output_bitmap;
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
    cudaMalloc(&d_input_bitmap, bitmap_len);
    cudaMalloc(&d_output_bitmap, bitmap_len);
    cudaMalloc(&d_status_bitmap, bitmap_size);
    cudaMalloc(&d_output_num, nf_num_size);
    cudaMalloc(&d_hops, hops_size);
    cudaMalloc(&d_update_frontiers, nf_size);
    // initial the input value
    cudaMemcpy(d_offset, offset.data(), offset_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, edges.data(), edge_size, cudaMemcpyHostToDevice);

    int index_in_bitmap, offset_in_bitmap;
    GET_BIT_INDEX_OFFSET(source, index_in_bitmap, offset_in_bitmap);
    int bitmask = 1 << offset_in_bitmap;
    for(int i = 0; i < bitmap_len; i++){
        bitmap[i] = 0;
    }
    bitmap[index_in_bitmap] = offset_in_bitmap;
    cudaMemcpy(d_input_bitmap, bitmap, bitmap_size, cudaMemcpyHostToDevice);
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

    //int sum = 0;
    int threadsPerBlock, blocksPerGrid;
    int cur_hop = 0;
    int nf_num = 1;
    calculate_kernel_config(nf_num, threadsPerBlock, blocksPerGrid);
    cudaEventRecord(start, 0);

    while(nf_num){
        //sum += nf_num;
        int smem_size = (2 * bitmap_len + (bitmap_len + blocksPerGrid - 1) / blockPerGrid*32) * sizeof(int);
        assert(smem_size < 48 * 1024);

        initial_output_bitmap<<<blocksPerGrid, threadsPerBlock, 0, stream_traversal>>>(d_output_bitmap, output_num, bitmap_len);
        get_output_frontiers_o1<<<blocksPerGrid, threadsPerBlock, smem_size, stream_traversal>>>(d_offset, d_edges, node_num, edge_num,
                                                                                d_input_bitmap, nf_num,
                                                                                d_output_bitmap, d_output_num,
                                                                                d_status_bitmap, bitmap_len,
                                                                                d_update_frontiers);

        cudaMemcpy(&output_num, d_output_num, nf_num_size, cudaMemcpyDeviceToHost);
        nf_num = output_num;
        //printf("nf_num: %d\n", nf_num);
        int* tmp = d_input_bitmap;
        d_input_bitmap = d_output_bitmap;
        d_output_bitmap = tmp;


        calculate_kernel_config(nf_num, threadsPerBlock, blocksPerGrid);
        update_node_status_o1<<<blocksPerGrid, threadsPerBlock, 0, stream_update>>>(d_update_frontiers, update_offset, nf_num,
                                                                    d_hops, cur_hop);
        update_offset += nf_num;
        cur_hop++;
        //printf("update_offset: %d\n", update_offset);
    }
    //printf("sum: %d\n", sum);
    cudaMemcpy(hops.data(), d_hops, hops_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU: bfs_hops_async_o1. Elapsed time :" << milliseconds << " (ms)\n";

    cudaFree(d_offset);
    cudaFree(d_edges);
    cudaFree(d_hops);
    cudaFree(d_input_bitmap);
    cudaFree(d_output_bitmap);
    cudaFree(d_status_bitmap);
    cudaFree(d_output_num);
    cudaFree(d_update_frontiers);

    cudaStreamDestroy(stream_traversal);
    cudaStreamDestroy(stream_update);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

std::vector<int> test_bfs_hops_async_o1(std::vector<int> offset, std::vector<int> endnodes, int source){
    int node_num = offset.size() - 1;
    int edge_num = endnodes.size();

    std::vector<int> hops(node_num, INVAILD);
    hops[source] = 0;
    // std::cout << "GPU:" << std::endl;

    bfs_hops_async_o1(offset, endnodes, node_num, edge_num, source, hops);

    //print_hops(source, hops);
    return hops;
}
