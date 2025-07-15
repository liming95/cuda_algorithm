#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>
#include "bfs_hops.cuh"

__global__ void get_output_frontiers_2(int* offset, int* edges, int node_num, int edge_num,
                                    int* input_frontiers, int nf_num,
                                    int* output_frontiers, int* output_num,
                                    int* status_bitmap, int bitmap_len);

__global__ void update_node_status_2(int* input_frontiers, int nf_num, int* hops, int hop);

void bfs_hops_async_2(std::vector<int> offset, std::vector<int> edges, int node_num, int edge_num,
                   int source,
                   std::vector<int> &hops);

inline void calculate_kernel_config(int thread_num, int& block_size, int& grid_size){
    int threadsPerBlock_up = ((thread_num + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    block_size = thread_num > BLOCK_MAX_SIZE ? BLOCK_MAX_SIZE : threadsPerBlock_up;
    grid_size = (thread_num + BLOCK_MAX_SIZE - 1) / BLOCK_MAX_SIZE;
}

__device__ int update_num;

__global__ void get_output_frontiers_2(int* offset, int* edges, int node_num, int edge_num,
                                    int* input_frontiers, int nf_num,
                                    int* output_frontiers, int* output_num,
                                    int* status_bitmap, int bitmap_len,
                                    int* update_frontiers){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int node, neighbor;
    int edge_start, edge_end;
    int index;
    int index_in_bitmap, offset_in_bitmap;

    if(tid == 0) {
        *output_num = 0;
        update_num += nf_num;
        //printf("update_num: %d\n", update_num);
    }

    //load bitmap into shared memory

    if(tid < nf_num){
        node = input_frontiers[tid];
        edge_start = offset[node];
        edge_end = offset[node+1];
        //printf("1:node: %d, its neighbors(%d) from %d to %d\n", node, edge_num, edge_start, edge_end);
        //Todo: memory coalesce, prefix sum, sorted order;
        for(int i = edge_start; i < edge_end; i++){
            // update status_bitmap
            neighbor = edges[i]; //printf("2.(%d, %d)\n", node, neighbor);
            
            GET_BIT_INDEX_OFFSET(neighbor, index_in_bitmap, offset_in_bitmap);
            int bitmark = 1 << offset_in_bitmap;
            int visited = atomicOr(&status_bitmap[index_in_bitmap], bitmark);
            //printf("3.node: %d, index:%d, offset:%d, status_bitmap: %x, visited_bitmap: %x\n", node, index_in_bitmap, offset_in_bitmap, status_bitmap[index_in_bitmap], visited);
            visited &= bitmark;

            // add neighbor into frontiers
            if(!visited) {
                index = atomicAdd(output_num, 1);
                output_frontiers[index] = neighbor;
                update_frontiers[update_num+index] = neighbor;
                //printf("4.neighbor: %d\n", neighbor);

                // printf("node %d: bitmap(%x)\n", neighbor, status_bitmap[index_in_bitmap]);
            }
        }
    }
}

__global__ void update_node_status_2(int* input_frontiers, int offset, int nf_num, int* hops, int hop){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if(tid == 0){
    //     printf("update_node_status_2: nf_num: %d\n", nf_num);
    // }

    if(tid < nf_num){
        int node = input_frontiers[offset+tid];
        //printf("node: %d, hop: %d\n", node, hops[node]); 
        hops[node] = hop + 1;
    }
}

void bfs_hops_async_2(std::vector<int> offset, std::vector<int> edges, int node_num, int edge_num,
                   int source,
                   std::vector<int> &hops){
    int* d_offset, *d_edges;
    int offset_size = offset.size() * sizeof(int);
    int edge_size = edge_num * sizeof(int);

    int* d_node_frontiers; //int* d_nf_num;
    int* d_output_frontiers; int* d_output_num; 
    int nf_size = node_num * sizeof(int);
    int nf_num_size = sizeof(int); // LANE_NUM 3

    int* d_hops;
    int hops_size = node_num * sizeof(int);


    int* d_status_bitmap;
    int word_bit_len = sizeof(int) * BYTE_SIZE;
    int bitmap_len = (node_num + word_bit_len - 1) / word_bit_len;
    int bitmap_size = bitmap_len * sizeof(int);

    int* bitmap = new int[bitmap_len];

    int* d_update_frontiers;

    int nf_num;
    int output_num = 0;
    int update_offset = 1;

    cudaMalloc(&d_offset, offset_size);
    cudaMalloc(&d_edges, edge_size);
    cudaMalloc(&d_node_frontiers, nf_size);
    cudaMalloc(&d_output_frontiers, nf_size);
    cudaMalloc(&d_status_bitmap, bitmap_size);
    //cudaMalloc(&d_nf_num, nf_num_size);
    cudaMalloc(&d_output_num, nf_num_size);
    cudaMalloc(&d_hops, hops_size);
    cudaMalloc(&d_update_frontiers, nf_size);

    cudaMemcpy(d_offset, offset.data(), offset_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, edges.data(), edge_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_frontiers, &source, sizeof(int), cudaMemcpyHostToDevice);
    nf_num = 1;
    bitmap[0] = 1;
    for(int i = 1; i < bitmap_len; i++){
        bitmap[i] = 0;
    }
    cudaMemcpy(d_status_bitmap, &bitmap[0], bitmap_size, cudaMemcpyHostToDevice);
    //cudaMemcpy(&d_nf_num, &nf_num, nf_num_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hops, hops.data(), hops_size, cudaMemcpyHostToDevice);

    int initial_value = 0;
    cudaMemcpyToSymbol(update_num, &initial_value, sizeof(int));

    cudaStream_t stream_traversal, stream_update;
    //cudaEvent_t nf_ready;
    int threadsPerBlock, blocksPerGrid;
    int cur_hop = 0;
    calculate_kernel_config(nf_num, threadsPerBlock, blocksPerGrid);

    cudaStreamCreate(&stream_traversal);
    cudaStreamCreate(&stream_update);
    //cudaEventCreate(&nf_ready);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    int sum = 0;
    while(nf_num){
        sum += nf_num;
        get_output_frontiers_2<<<blocksPerGrid, threadsPerBlock, 0, stream_traversal>>>(d_offset, d_edges, node_num, edge_num,
                                                                                d_node_frontiers, nf_num,
                                                                                d_output_frontiers, d_output_num,
                                                                                d_status_bitmap, bitmap_len,
                                                                                d_update_frontiers);
        //cudaEventRecord(nf_ready, stream_traversal);
        cudaMemcpy(&output_num, d_output_num, nf_num_size, cudaMemcpyDeviceToHost);
        nf_num = output_num;
        //printf("nf_num: %d\n", nf_num);
        int* tmp = d_node_frontiers;
        d_node_frontiers = d_output_frontiers;
        d_output_frontiers = tmp;

        //cudaStreamWaitEvent(nf_ready, stream_update);
        calculate_kernel_config(nf_num, threadsPerBlock, blocksPerGrid);
        update_node_status_2<<<blocksPerGrid, threadsPerBlock, 0, stream_update>>>(d_update_frontiers, update_offset, nf_num,
                                                                   d_hops, cur_hop);
        update_offset += nf_num;
        cur_hop++;
        //printf("update_offset: %d\n", update_offset);
    }
    //printf("sum: %d\n", sum);
    cudaMemcpy(hops.data(), d_hops, hops_size, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < hops.size(); i++){
    //     std::cout << "hops[" << i << "]: " << hops << std::endl;
    // }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU: bfs_hops_async_update. Elapsed time :" << milliseconds << " (ms)\n";

    cudaFree(d_offset);
    cudaFree(d_edges);
    cudaFree(d_hops);
    cudaFree(d_node_frontiers);
    cudaFree(d_output_frontiers);
    cudaFree(d_status_bitmap);
    cudaFree(d_output_num);
    cudaFree(d_update_frontiers);

    cudaStreamDestroy(stream_traversal);
    cudaStreamDestroy(stream_update);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

std::vector<int> test_bfs_hops_async_2(std::vector<int> offset, std::vector<int> endnodes, int source){
    int node_num = offset.size() - 1;
    int edge_num = endnodes.size();

    std::vector<int> hops(node_num, INVAILD);
    hops[source] = 0;
    // std::cout << "GPU:" << std::endl;

    bfs_hops_async_2(offset, endnodes, node_num, edge_num, source, hops);

    //print_hops(source, hops);
    return hops;
}
