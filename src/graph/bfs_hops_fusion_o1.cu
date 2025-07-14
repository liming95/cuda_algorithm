#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>

#include "bfs_hops.cuh"


__device__ int vf_num, block_offset, producer_num, processed_num;

/* opt 1: load balance by binary search in one blk
 * opt 2: deliver frontiers in different
 * opt 3: load balance in differen block
*/
__global__ void bfs_hops_fusion_2_o1(int* g_offset, int* g_edges, int node_num, int edge_num,
                              int* input_frontiers, int input_num,
                              int* vertex_frontiers,
                              int* output_frontiers, int* output_num,
                              // int* visited_bitmap, // bitmap
                              int* distance,
                              int next_hop
                            ){
    int vertex, cur_hop;
    int start, end;
    int neighbors_num[1];
    int neighbor, ngb_hop;
    int hop = next_hop;
    int input_num_reg = input_num;

    int glb_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blk_tid = threadIdx.x;
    int block_size = blockDim.x;  // threads per block
    int grid_size = gridDim.x;    // blocks per grid

    int total_edges_per_block;
    int degrees_length;

    typedef cub::BlockScan<int, BLOCK_MAX_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ int blk_degrees[BLOCK_MAX_SIZE];
    __shared__ int blk_vertexs[BLOCK_MAX_SIZE];
    __shared__ int blk_start_offset[BLOCK_MAX_SIZE];
    __shared__ int blk_vertex_frontiers[BLOCK_MAX_SIZE];

    blk_vertex_frontiers[blk_tid] = 0;

    // First Level
    if(glb_tid < input_num_reg){
      vertex = input_frontiers[glb_tid];
      blk_vertexs[blk_tid] = vertex;
      cur_hop = distance[vertex];         // Todo: eliminate the distance access
      DEBUG_PRINT("(1)vertex: %d, hop: %d\n", vertex, cur_hop);

      //Todo: test performance when there is no filtering operation.
      // filter the non-frontier vertex
      if((cur_hop & ODD) == EVEN) {
        start = g_offset[vertex];
        end   = g_offset[vertex+1];       // Todo: eliminate the end access
        blk_start_offset[blk_tid] = start;
        neighbors_num[0] = end - start;
        DEBUG_PRINT("(1)effective vertex: %d, ngh_num: %d\n", vertex, neighbors_num[0]);
      }
      else {
        neighbors_num[0] = 0;
      }
    }
    else {
      neighbors_num[0] = 0;
    }

    BlockScan(temp_storage).ExclusiveSum(neighbors_num, neighbors_num, total_edges_per_block);

    blk_degrees[blk_tid] = neighbors_num[0];
    __syncthreads();
    //DEBUG_PRINT("thread:(%d, %d), prefix_sum: %d, total_ngb: %d\n", glb_tid, blk_tid, blk_degrees[blk_tid], total_edges_per_block);


    degrees_length = blockIdx.x < (grid_size - 1) ? block_size : input_num_reg - (glb_tid - blk_tid);
    //binary search
    for(int i = blk_tid; i < total_edges_per_block; i += block_size){
      auto it = thrust::upper_bound(thrust::seq, blk_degrees, blk_degrees+degrees_length, i);
      int idx = thrust::distance(blk_degrees, it) - 1;

      vertex = blk_vertexs[idx];
      start = blk_start_offset[idx];
      int offset_in_ngb_per_vertex = i - blk_degrees[idx];
      neighbor = g_edges[start+offset_in_ngb_per_vertex];

      ngb_hop = atomicMin(&distance[neighbor], hop);
      DEBUG_PRINT("(1)neighbor:(%d, %d), pre_hop: %d, hop: %d\n", vertex, neighbor, ngb_hop, hop);

      // produce frontiers for next step
      if(hop < ngb_hop) {
        // frontiers in blk
        vertex = blk_vertex_frontiers[blk_tid];
        bool filled = vertex == 0;
        blk_vertex_frontiers[blk_tid] = filled ? neighbor : vertex;

        // in global pool
        if(!filled) {
          int index = atomicAdd(&vf_num, 1);
          vertex_frontiers[index] = neighbor;
        }
      }
    }
    __syncthreads();

    // Second Level
    // 1. consume the frontiers in block
    hop += 1;
    if(blk_vertex_frontiers[blk_tid] != 0) {
      vertex = blk_vertex_frontiers[blk_tid];
      start = g_offset[vertex];
      end = g_offset[vertex+1];
      blk_start_offset[blk_tid] = start;
      neighbors_num[0] = end - start;
      DEBUG_PRINT("(2.1)vertex: %d, ngb_num: %d\n", vertex, neighbors_num[0]);
    }
    else{
      neighbors_num[0] = 0;
    }
    BlockScan(temp_storage).ExclusiveSum(neighbors_num, neighbors_num, total_edges_per_block);

    blk_degrees[blk_tid] = neighbors_num[0];
    __syncthreads();
    //DEBUG_PRINT("(2.1)thread:(%d, %d), prefix_sum: %d, total_ngb: %d\n", glb_tid, blk_tid, blk_degrees[blk_tid], total_edges_per_block);

    for(int i = blk_tid; i < total_edges_per_block; i += block_size){
      auto it = thrust::upper_bound(thrust::seq, blk_degrees, blk_degrees+block_size, i);
      int idx = thrust::distance(blk_degrees, it) - 1;
      DEBUG_PRINT("(2.1) i: %d, idx: %d\n", i, idx);
      vertex = blk_vertex_frontiers[idx];
      start = blk_start_offset[idx];
      int offset_in_ngb_per_vertex = i - blk_degrees[idx];
      neighbor = g_edges[start+offset_in_ngb_per_vertex];

      ngb_hop = atomicMin(&distance[neighbor], hop);
      DEBUG_PRINT("(2.1)neighbor:(%d, %d), pre_hop: %d, hop: %d\n", vertex, neighbor, ngb_hop, hop);
      // produce frontiers for next step
      if(hop < ngb_hop) {
        int index = atomicAdd(output_num, 1);
        output_frontiers[index] = neighbor;
      }
    }
    __syncthreads();

    // 2. consume the remainding vertex in glb task pool
    __shared__ int task_size, task_offset;
    if(threadIdx.x == 0){
      atomicAdd(&producer_num, 1);
      task_offset = atomicAdd(&block_offset, block_size);
    }

    while((producer_num < grid_size) || (processed_num < vf_num)){
      // load task
      if(threadIdx.x == 0){
        if(vf_num > task_offset){
          int offset = task_offset + block_size;
          int vertex_num = atomicMax(&vf_num, offset);
          task_size = vertex_num > offset ? block_size : vertex_num - task_offset;
        } else {
          task_size = 0;
        }
        //printf("(2)task_size: %d, task_offset: %d, block_size: %d, vf_num: %d\n",task_size, task_offset, block_size, vertex_num);
      }
      __syncthreads();

      input_num_reg = task_size;
      if (input_num_reg == 0) continue;

      // process remainding task

      if(blk_tid < input_num_reg) {
        vertex = vertex_frontiers[blk_tid];
        blk_vertexs[blk_tid] = vertex;
        start = g_offset[vertex];
        end = g_offset[vertex];
        neighbors_num[0] = end - start;
        blk_start_offset[blk_tid] = start;
        DEBUG_PRINT("(2.2)vertex: %d, hop: %d\n", vertex, cur_hop);

      }
      else {
        neighbors_num[0] = 0;
      }
      BlockScan(temp_storage).ExclusiveSum(neighbors_num, neighbors_num, total_edges_per_block);

      blk_degrees[blk_tid] = neighbors_num[0];
      __syncthreads();

      for(int i = blk_tid; i < total_edges_per_block; i += block_size){
        auto it = thrust::upper_bound(thrust::seq, blk_degrees, blk_degrees+input_num_reg, i);
        int idx = thrust::distance(blk_degrees, it) - 1;

        vertex = blk_vertexs[idx];
        start = blk_start_offset[idx];
        int offset_in_ngb_per_vertex = i - blk_degrees[idx];
        neighbor = g_edges[start+offset_in_ngb_per_vertex];

        ngb_hop = atomicMin(&distance[neighbor], hop);
        // produce frontiers for next step
        if(hop < ngb_hop) {
          int index = atomicAdd(output_num, 1);
          output_frontiers[index] = neighbor;
        }
      }
      __syncthreads();

      if((threadIdx.x == 0) && (input_num_reg != 0)){
        atomicAdd(&processed_num, block_size);
        task_offset = atomicAdd(&block_offset, block_size);
      }
    }

    if(glb_tid == 0) {
      vf_num = 0;
      block_offset = 0;
      producer_num = 0;
      processed_num = 0;
    }
}

std::vector<int> test_bfs_hops_fusion_o1(std::vector<int> offset, std::vector<int> edges, int source){
    // build queue
    int * queue_in;
    // * queue_out;
    int node_num = offset.size() - 1;
    int edge_num = edges.size();
    int queue_in_num = 0, queue_out_num = 0;

    queue_in = new int[node_num];
    //queue_out = new int[node_num];
    queue_in[0] = source;
    queue_in_num++;

    // initial hops
    std::vector<int> hops(node_num, INVAILD);
    hops[source] = 0;

    // copy to device
    int queue_num_size = sizeof(int);
    int offset_size = offset.size() * sizeof(int);
    int edges_size = edge_num * sizeof(int);
    int queue_size = node_num * sizeof(int);
    int hops_size = node_num * sizeof(int);
    int* d_offset, *d_edges, *d_queue_in, *d_queue_out, *d_hops, *d_queue_out_num;
    int* d_vertex_frontiers;
    int* d_vf_num;

    cudaMalloc(&d_offset, offset_size);
    cudaMalloc(&d_edges, edges_size);
    cudaMalloc(&d_queue_in, queue_size);
    cudaMalloc(&d_queue_out, queue_size);
    cudaMalloc(&d_hops, hops_size);
    //cudaMalloc(&d_queue_in_num, queue_num_size);
    cudaMalloc(&d_queue_out_num, queue_num_size);
    cudaMalloc(&d_vertex_frontiers, queue_size);
    cudaMalloc(&d_vf_num, queue_num_size);

    cudaMemcpy(d_offset, offset.data(), offset_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, edges.data(), edges_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_queue_in, queue_in, queue_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hops, hops.data(), hops_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_queue_out_num, &queue_out_num, queue_num_size, cudaMemcpyHostToDevice);

    int initial_value = 0;
    cudaMemcpyToSymbol(vf_num, &initial_value, sizeof(int));
    cudaMemcpyToSymbol(producer_num, &initial_value, sizeof(int));
    cudaMemcpyToSymbol(block_offset, &initial_value, sizeof(int));
    cudaMemcpyToSymbol(processed_num, &initial_value, sizeof(int));
    // std::cout << "GPU no perf\n";
    // int loop = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    int next_hop = 1;
    do {
        // block size
        // int threadsPerBlock_up = ((queue_in_num + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        // int block_size = queue_in_num > BLOCK_MAX_SIZE ? BLOCK_MAX_SIZE : threadsPerBlock_up;
        int block_size = BLOCK_MAX_SIZE;
        int grid_size = (queue_in_num + BLOCK_MAX_SIZE - 1) / BLOCK_MAX_SIZE;

        int threadsPerBlock = block_size;
        int blocksPerGrid = grid_size;
        // std::cout << "current hop (" << loop << "): queue_in_num (" << queue_in_num << ")\n";

        // kernel launch

        bfs_hops_fusion_2_o1<<<blocksPerGrid, threadsPerBlock>>>(d_offset, d_edges, offset.size(), edge_num,
                                                              d_queue_in, queue_in_num,
                                                              d_vertex_frontiers,
                                                              d_queue_out, d_queue_out_num,
                                                              d_hops, next_hop);

        CHECK_CUDA_SYNC("After device synchronize");

        // copy queue_out_num to host
        cudaMemcpy(&queue_out_num, d_queue_out_num, queue_num_size, cudaMemcpyDeviceToHost);
        queue_in_num = queue_out_num;
        queue_out_num = 0;
        cudaMemcpy(d_queue_out_num, &queue_out_num, queue_num_size, cudaMemcpyHostToDevice);
        int * tmp;
        tmp = d_queue_in;
        d_queue_in = d_queue_out;
        d_queue_out = tmp;
        next_hop += 2;
    }
    while(queue_in_num);

    // copy hops from device to host
    cudaMemcpy(hops.data(), d_hops, hops_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU: bfs_hops_fusion_o1. Elapsed time :" << milliseconds << " (ms)\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_offset);
    cudaFree(d_edges);
    cudaFree(d_queue_in);
    cudaFree(d_queue_out);
    cudaFree(d_hops);
    cudaFree(d_queue_out_num);

    //print_hops(source, hops);

    return hops;
}
