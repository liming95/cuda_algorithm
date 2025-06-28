#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>
#include "bfs_hops.cuh"

__global__ void get_edge_frontiers(int* offset, int* edges, int node_num, int edge_num,
                                    int* node_frontiers, int* nf_num,
                                    int* edge_frontiers, int* ef_num,
                                    int* edges_bitmap, int* edges_bitmap_prev, int bitmap_len);

__global__ void get_node_frontiers(int* edges_bitmap, int bitmap_len,
                                    int* edge_frontiers, int* ef_num,
                                    int* node_frontiers, int* nf_num,
                                    int* node_frontiers_status);

__global__ void updata_node_status(int* node_frontiers, int nf_num, int* hops, int hop);

void bfs_hops_async(std::vector<int> offset, std::vector<int> edges, int node_num, int edge_num,
                   int source,
                   std::vector<int> &hops);

inline void calculate_kernel_config(int thread_num, int& block_size, int& grid_size){
    int threadsPerBlock_up = ((thread_num + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    block_size = thread_num > BLOCK_MAX_SIZE ? BLOCK_MAX_SIZE : threadsPerBlock_up;
    grid_size = (thread_num + BLOCK_MAX_SIZE - 1) / BLOCK_MAX_SIZE;
}
#define GET_BIT_INDEX_OFFSET(pos, index, offset) \
    int word_bit_len = (sizeof(int)) * BYTE_SIZE; \
    index = pos / word_bit_len; \
    offset = pos % word_bit_len

__global__ void get_edge_frontiers(int* offset, int* edges, int node_num, int edge_num,
                                    int* node_frontiers, int* nf_num,
                                    int* edge_frontiers, int* ef_num,
                                    int* edges_bitmap, int* edges_bitmap_prev, int bitmap_len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int node, neighbor;
    if(tid == 0) {
        *ef_num = 0;
    }
    int thread_num = gridDim.x * blockDim.x;
    int iter = bitmap_len / thread_num;
    int remaind = bitmap_len % thread_num;
    int cursor = 0;
    int base, index;
    while(cursor < iter) {
        base = cursor * thread_num;
        index = base + tid;
        edges_bitmap[index] |= edges_bitmap_prev[index];
        cursor++;
    }
    base = iter * thread_num;
    if(tid < remaind){
        index = base + tid;
        edges_bitmap[index] |= edges_bitmap_prev[index];
    }

    if(tid < *nf_num){
        node = node_frontiers[tid];
        int edge_start = offset[node];
        int edge_end = offset[node+1];
        // int edge_num = edge_end - edge_start;
        // int index = atomicAdd(ef_num, edge_num);
        // printf("node: %d, its neighbors(%d) from %d to %d\n", node, edge_num, edge_start, edge_end);
        //Todo: memory coalesce, prefix sum, sorted order;
        for(int i = edge_start; i < edge_end; i++){
            // update edges_bitmap
            neighbor = edges[i]; //printf("(%d, %d)\n", node, neighbor);
            int index_in_bitmap, offset_in_bitmap;
            GET_BIT_INDEX_OFFSET(neighbor, index_in_bitmap, offset_in_bitmap);
            int bitmark = 1 << offset_in_bitmap;
            int visited = atomicOr(&edges_bitmap[index_in_bitmap], bitmark);
            visited &= bitmark;

            // add neighbor into frontiers
            if(!visited) {
                index = atomicAdd(ef_num, 1);
                edge_frontiers[index] = neighbor;
                // printf("node %d: bitmap(%x)\n", neighbor, edges_bitmap[index_in_bitmap]);
            }
        }
    }
}

__global__ void get_node_frontiers(int* edges_bitmap, int bitmap_len,
                                    int* edge_frontiers, int* ef_num,
                                    int* node_frontiers, int* nf_num,
                                    int* node_frontiers_status){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid == 0){
        *nf_num = 0;
    }
    if(tid < *ef_num){
        int node = edge_frontiers[tid];
        // int index, offset;
        // GET_BIT_INDEX_OFFSET(node, index, offset);
        // assert(index < bitmap_len);

        // int visited = edges_bitmap[index];
        // int bitmask = 1 << offset;
        // visited &= bitmask;

        // if(!visited){
        //     //add to node_frontiers
        //     printf("no visited node: %d, bitmap: (%d, %d) %d\n", node, index, offset, visited);
        //     index = atomicAdd(nf_num, 1);
        //     node_frontiers[index] = node;
        //     node_frontiers_status[index] = node;
        // } else {
        //     printf("visited node: %d, bitmap: (%d, %d) %d\n", node, index, offset, visited);
        // }
        *nf_num = *ef_num;
        node_frontiers[tid] = node;
        node_frontiers_status[tid] = node;
    }
}

__global__ void updata_node_status(int* node_frontiers, int nf_num, int* hops, int hop){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < nf_num){
        int node = node_frontiers[tid];
        hops[node] = hop + 1;
    }
}

void bfs_hops_async(std::vector<int> offset, std::vector<int> edges, int node_num, int edge_num,
                   int source,
                   std::vector<int> &hops){
    int* d_offset, *d_edges;
    int offset_size = offset.size() * sizeof(int);
    int edge_size = edge_num * sizeof(int);

    int* d_node_frontiers[LANE_NUM]; int* d_nf_num[LANE_NUM];
    int* d_edge_frontiers[LANE_NUM]; int* d_ef_num[LANE_NUM];
    int* d_node_frontiers_status[LANE_NUM];
    int nf_size = node_num * sizeof(int);
    int ef_size = edge_num * sizeof(int);
    int nf_num_size = sizeof(int); // LANE_NUM 3

    int* d_hops;
    int hops_size = node_num * sizeof(int);


    int* d_edges_bitmap[LANE_NUM];
    int word_bit_len = sizeof(int) * BYTE_SIZE;
    int bitmap_len = (node_num + word_bit_len - 1) / word_bit_len;
    int bitmap_size = bitmap_len * sizeof(int);

    int* bitmap = new int[bitmap_len];

    int nf_num[LANE_NUM] = {0};
    int ef_num[LANE_NUM] = {0};

    cudaMalloc(&d_offset, offset_size);
    cudaMalloc(&d_edges, edge_size);
    for(int i = 0; i < LANE_NUM; i++){
        cudaMalloc(&d_node_frontiers[i], nf_size);
        cudaMalloc(&d_node_frontiers_status[i], nf_size);
        cudaMalloc(&d_edge_frontiers[i], ef_size);
        cudaMalloc(&d_edges_bitmap[i], bitmap_size);
        cudaMalloc(&d_nf_num[i], nf_num_size);
        cudaMalloc(&d_ef_num[i], nf_num_size);
    }

    cudaMalloc(&d_hops, hops_size);

    cudaMemcpy(d_offset, offset.data(), offset_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, edges.data(), edge_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_frontiers[0], &source, sizeof(int), cudaMemcpyHostToDevice);
    nf_num[0] = 1;
    bitmap[0] = 1;
    for(int i = 1; i < bitmap_len; i++){
        bitmap[i] = 0;
    }

    for(int i = 0; i < LANE_NUM; i++){
        cudaMemcpy(d_nf_num[i], &nf_num[i], nf_num_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_edges_bitmap[i], bitmap, bitmap_size, cudaMemcpyHostToDevice);
    }


    cudaMemcpy(d_hops, hops.data(), hops_size, cudaMemcpyHostToDevice);

    cudaStream_t streams[LANE_NUM];
    cudaEvent_t edge_bitmap_ready[LANE_NUM];
    cudaEvent_t nf_ready[LANE_NUM];

    for(int i = 0; i < LANE_NUM; i++){
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&edge_bitmap_ready[i]);
        cudaEventCreate(&nf_ready[i]);
    }

    bool has_node_frontier[LANE_NUM] = {true, false, false};
    bool has_edge_frontier[LANE_NUM] = {false, false, false};

    cudaEventRecord(nf_ready[0], streams[0]);
    cudaEventRecord(edge_bitmap_ready[LANE_NUM-1], streams[LANE_NUM-1]);

    bool has_task = has_node_frontier[0] || has_node_frontier[1] || has_node_frontier[2];
    int loop = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    while(has_task) {
        // std::cout << "The " << loop << " iterate\n";
        for(int i = 0; i < LANE_NUM; i++){
            int cur_hop = loop * LANE_NUM + i;
            // std::cout << "The " << i << " lane: current hop (" << cur_hop << ")\n";
            int prev = (i + 2) % LANE_NUM;
            int next = (i + 1) % LANE_NUM;
            cudaStream_t stream = streams[i];
            int threadsPerBlock, blocksPerGrid;

            if(has_node_frontier[i]){
                cudaStreamWaitEvent(stream, nf_ready[i], 0);
                // block size
                calculate_kernel_config(nf_num[i], threadsPerBlock, blocksPerGrid);
                //stage 1: generate neighbors
                get_edge_frontiers<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_offset, d_edges, node_num, edge_num,
                                                                                d_node_frontiers[i], d_nf_num[i],
                                                                                d_edge_frontiers[i], d_ef_num[i],
                                                                                d_edges_bitmap[i], d_edges_bitmap[prev], bitmap_len);

                cudaEventRecord(edge_bitmap_ready[i], stream);
                has_node_frontier[i] = false;
                has_edge_frontier[i] = true;
                // std::cout << "stage 1:" << "nf_num (" << nf_num[i] << ")\n";
            }

            if(!has_node_frontier[prev] && has_edge_frontier[i]) {
                cudaStreamWaitEvent(stream, edge_bitmap_ready[prev], 0);
                cudaMemcpy(&ef_num[i], d_ef_num[i], nf_num_size, cudaMemcpyDeviceToHost);
                // std::cout << "stage2: ef_num (" << ef_num[i] << ")\n";
                if(ef_num[i] != 0){
                    //stage 2: filter the visited node according to the previous edge bitmap and visited bitmap
                    calculate_kernel_config(ef_num[i], threadsPerBlock, blocksPerGrid);
                    get_node_frontiers<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_edges_bitmap[prev], bitmap_len,
                                                                                    d_edge_frontiers[i], d_ef_num[i],
                                                                                    d_node_frontiers[next], d_nf_num[next],
                                                                                    d_node_frontiers_status[i]);
                    cudaEventRecord(nf_ready[next], stream);
                    has_edge_frontier[i] = false;

                    cudaMemcpy(&nf_num[next], d_nf_num[next], nf_num_size, cudaMemcpyDeviceToHost);
                    int nf_num_status = nf_num[next];
                    //stage 3: update the status array for current visited nodes.
                    if(nf_num_status != 0){
                        has_node_frontier[next] = true;
                        calculate_kernel_config(nf_num_status, threadsPerBlock, blocksPerGrid);
                        updata_node_status<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_node_frontiers_status[i], nf_num_status,
                                                                                    d_hops, cur_hop);
                        // std::cout << "stage 3: nf_num (" << nf_num_status << ")\n";
                    }

                }
            }
        }
        loop++;
        has_task = has_node_frontier[0] || has_node_frontier[1] || has_node_frontier[2];
    }
    cudaMemcpy(hops.data(), d_hops, hops_size, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < hops.size(); i++){
    //     std::cout << "hops[" << i << "]: " << hops[i] << std::endl;
    // }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU: bfs_hops_async. Elapsed time :" << milliseconds << " (ms)\n";

    cudaFree(d_offset);
    cudaFree(d_edges);
    cudaFree(d_hops);
    for(int i = 0; i < LANE_NUM; i++){
        //cudaStreamSynchronize(streams[i]);
        cudaEventDestroy(edge_bitmap_ready[i]);
        cudaEventDestroy(nf_ready[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(d_node_frontiers[i]);
        cudaFree(d_node_frontiers_status[i]);
        cudaFree(d_edge_frontiers[i]);
        cudaFree(d_edges_bitmap[i]);
        cudaFree(d_nf_num[i]);
        cudaFree(d_ef_num[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

std::vector<int> test_bfs_hops_async(std::vector<int> offset, std::vector<int> endnodes, int source){
    int node_num = offset.size() - 1;
    int edge_num = endnodes.size();

    std::vector<int> hops(node_num, INVAILD);
    hops[source] = 0;
    // std::cout << "GPU:" << std::endl;

    bfs_hops_async(offset, endnodes, node_num, edge_num, source, hops);

    //print_hops(source, hops);
    return hops;
}
