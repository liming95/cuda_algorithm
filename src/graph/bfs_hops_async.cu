#include <iostream>
#include <cuda_runtime.h>
#include "bfs_hops.cuh"

__global__ void get_edge_frontiers(int* offset, int* edges, int node_num, int edge_num,
                                    int* node_frontiers, int* nf_num,
                                    int* edge_frontiers, int* ef_num,
                                    int* edges_bitmap);

__global__ void get_node_frontiers(int* edges_bitmap, int* visited_bitmap, int bitmap_len,
                                    int* edge_frontiers, int* ef_num,
                                    int* node_frontiers, int* nf_num,
                                    int* node_frontiers_status);

__global__ void updata_node_status(int* node_frontiers, int nf_num, int* hops, int* visited_bitmap);

void bfs_hops_async(std::vector<int> offset, std::vector<int> edges, int node_num, int edge_num,
                   int source,
                   std::vector<int> hops);

inline void calculate_kernel_config(int thread_num, int block_size, int grid_size){
    int threadsPerBlock_up = ((thread_num + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    block_size = thread_num > BLOCK_MAX_SIZE ? BLOCK_MAX_SIZE : threadsPerBlock_up;
    grid_size = (thread_num + BLOCK_MAX_SIZE - 1) / BLOCK_MAX_SIZE;
}

void bfs_hops_async(std::vector<int> offset, std::vector<int> edges, int node_num, int edge_num,
                   int source,
                   std::vector<int> hops){
    int* d_offset, *d_edges;
    int* d_node_frontiers[LANE_NUM];
    int* d_node_frontiers_status[LANE_NUM];
    int* d_edge_frontiers[LANE_NUM];
    int nf_num[LANE_NUM] = {0};
    int ef_num[LANE_NUM] = {0};
    int* d_nf_num[LANE_NUM];
    int* d_ef_num[LANE_NUM];
    int* d_edges_bitmap[LANE_NUM];
    int* d_visited_bitmap;
    int* d_hops;

    int offset_size = node_num * sizeof(int);
    int edge_size = edge_num * sizeof(int);
    int nf_size = node_num * sizeof(int);
    int ef_size = edge_num * sizeof(int);
    int hops_size = node_num * sizeof(int);
    int word_size = sizeof(int) * BYTE_SIZE;
    int bitmap_len = (node_num + bit_len - 1) / word_size
    int bitmap_size = bitmap_len * sizeof(int);
    int* bitmap = new int[bitmap_len];
    int nf_num_size = sizeof(int); // LANE_NUM 3

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

    cudaMalloc(&d_visited_bitmap, bitmap_size);
    cudaMalloc(&d_hops, hops_size);

    cudaMemcpy(d_offset, offset.data(), offset_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, edges.data(), edge_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_frontiers[0], &source, sizeof(int), cudaMemcpyHostToDevice);
    nf_num[0] = 1;

    for(int i = 0; i < LANE_NUME; i++){
        cudaMemcpy(d_nf_num[i], &nf_num[i], nf_num_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_edges_bitmap[i], bitmap, bitmap_size, cudaMemcpyHostToDevice);
    }

    bitmap[0] = 1;
    cudaMemcpy(d_visited_bitmap, bitmap, bitmap_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hops, hops, hops_size, cudaMemcpyHostToDevice);

    cudaStream_t streams[LANE_NUM];
    cudaEvent_t edge_bitmap_ready[LANE_NUM];
    cudaEvent_t nf_ready[LANE_NUM];

    bool has_node_frontier[LANE_NUM] = {true, false, false};
    bool has_edge_frontier[LANE_NUM] = {false, false, false};

    cudaEventRecord(nf_ready[0], streams[0]);
    cudaEventRecord(edge_bitmap_ready[LANE_NUM], streams[LANE_NUM]);

    bool has_task = has_node_frontier[0] || has_node_frontier[1] || has_node_frontier[2];
    while(has_task) {
        for(int i = 0; i < LANE_NUM; i++){
            int prev = (i + 2) % LANE_NUM;
            int next = (i + 1) % LANE_NUM;
            cudaStream_t stream = streams[i];
            int threadsPerBlock, blocksPerGrid;

            if(has_node_frontier[i]){
                cudaStreamWaitEvent(stream, nf_ready[i], 0);
                // block size
                calculate_kernel_config(nf_num[i], threadsPerBlock, blocksPerGrid);
                //stage 1: generate neighbors
                get_edge_frontiers<<<blocksPerGrid, threadsPerBlock, stream>>>(d_offset, d_edges, node_num, edge_num,
                                                                                d_node_frontiers[i], d_nf_num[i],
                                                                                d_edge_frontiers[i], d_ef_num[i],
                                                                                d_edges_bitmap[i]);

                cudaEventRecord(edge_bitmap_ready[i], stream);
                has_node_frontier[i] = false;
            }

            if(!has_node_frontier[prev]) {
                cudaStreamWaitEvent(stream, edge_bitmap_ready[prev], 0);
                cudaMemcpy(&ef_num[i], d_ef_num[i], nf_num_size, cudaMemcpyDeviceToHost);
                if(ef_num[i] != 0){
                    //stage 2: filter the visited node according to the previous edge bitmap and visited bitmap
                    calculate_kernel_config(ef_num[i], threadsPerBlock, blocksPerGrid);
                    get_node_frontiers<<<blocksPerGrid, threadsPerBlock, stream>>>(d_edges_bitmap[prev], d_visited_bitmap, bitmap_len
                                                                                    d_edge_frontiers[i], d_ef_num[i],
                                                                                    d_node_frontiers[next], d_nf_num[i],
                                                                                    d_node_frontiers_status[i]);
                    cudaEventRecord(node_ready[next], stream);
                    has_node_frontier[next] = true;

                    cudaMemcpy(&nf_num[i], d_nf_num[i], nf_num_size, cudaMemcpyDeviceToHost);

                    //stage 3: update the status array for current visited nodes.
                    calculate_kernel_config(nf_num[i], threadsPerBlock, blocksPerGrid);
                    updata_node_status<<<blocksPerGrid, threadsPerBlock, stream>>>(d_node_frontiers_status[i], nf_num[i],
                                                                                    d_hops, d_visited_bitmap);
                }


            }
        }
    }


}
