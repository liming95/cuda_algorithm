#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>
#include "bfs_hops.cuh"


__device__ int vf_num, block_offset, producer_num, processed_num;

__global__ void bfs_hops_fusion_2(int* g_offset, int* g_edges, int node_num, int edge_num,
                              int* input_frontiers, int input_num,
                              int* vertex_frontiers,
                              int* output_frontiers, int* output_num,
                              // int* visited_bitmap, // bitmap
                              int* distance,
                              int next_hop
                            ){
    int vertex, cur_hop;
    int start, end;
    int neighbor, n_hop;
    int index;
    int vertex_num;
    __shared__ int task_size, task_offset;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blk_tid = threadIdx.x;
    int block_size = blockDim.x;
    int tmp_hop;
    int offset;

    if(tid == 0) {
      vf_num = 0;
      block_offset = 0;
      producer_num = 0;
      processed_num = 0;
    }
    if(tid < input_num){
      vertex = input_frontiers[tid];
      // int index = vertex / INT_BIT_LEN;
      // int offset = vertex % INT_BIT_LEN;
      // int mask = 1 << offset;
      // int cur_hop = visited_bitmap[index] & mask == 0 ? distance[vertex] : ODD;

      // 1. filtering the input data
      cur_hop = distance[vertex];
      //printf("(1)vertex: %d, hop: %d\n", vertex, cur_hop);
      if((cur_hop & ODD) == EVEN) {
        //cur_hop += 1;
        // 2. expanding neighbors
        start = g_offset[vertex];
        end   = g_offset[vertex+1];
        // 3. update neighbors & add into next input and edge forntiers
        for(int i = start; i < end; i++){
          neighbor = g_edges[i];
          //printf("(1)neighbor: %d\n", neighbor);
          n_hop = atomicMin(&distance[neighbor], next_hop);
          if(next_hop < n_hop){
            //printf("(1)enqueue:%d\n", neighbor);
            index = atomicAdd(&vf_num, 1);
            vertex_frontiers[index] = neighbor;
            //assert(vf_num <= node_num);
          }
        }
      }
    }
    __syncthreads();
    // 4. calculate the dynamically generated work.
    if(threadIdx.x == 0){
      atomicAdd(&producer_num, 1);
      task_offset = atomicAdd(&block_offset, block_size);
    }
    while((producer_num < gridDim.x) || (processed_num < vf_num)){
      // load task
      if(threadIdx.x == 0){
        if(vf_num > task_offset){
          offset = task_offset + block_size;
          vertex_num = atomicMax(&vf_num, offset);
          task_size = vertex_num > offset ? block_size : vertex_num - task_offset;
        } else {
          task_size = 0;
        }
        //printf("(2)task_size: %d, task_offset: %d, block_size: %d, vf_num: %d\n",task_size, task_offset, block_size, vertex_num);
      }
       __syncthreads();
      //
      // calculate hops
      if(blk_tid < task_size){
        tmp_hop = next_hop + 1;
        //printf("(2)blk_tid: %d, task_offset: %d\n",blk_tid, task_offset);
        index = blk_tid + task_offset;
        vertex = vertex_frontiers[index];
        //printf("(2)blk_tid: %d, dequeue_vertex: %d\n",blk_tid, vertex);

        // expanding neighbors
        start = g_offset[vertex];
        end   = g_offset[vertex+1];
        // update neighbors & add into next input and edge forntiers
        for(int i = start; i < end; i++){
          neighbor = g_edges[i];
          n_hop = atomicMin(&distance[neighbor], tmp_hop);
          if(tmp_hop < n_hop){
            index = atomicAdd(output_num, 1);
            output_frontiers[index] = neighbor;
          }
        }
      }
      __syncthreads();
      if((threadIdx.x == 0) && (task_size != 0)){
        atomicAdd(&processed_num, block_size);
        task_offset = atomicAdd(&block_offset, block_size);
      }
    }
}
std::vector<int> test_bfs_hops_fusion(std::vector<int> offset, std::vector<int> edges, int source){
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
    // std::cout << "GPU no perf\n";
    // int loop = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    int next_hop = 1;
    do {
        // block size
        int threadsPerBlock_up = ((queue_in_num + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        int block_size = queue_in_num > BLOCK_MAX_SIZE ? BLOCK_MAX_SIZE : threadsPerBlock_up;
        int grid_size = (queue_in_num + BLOCK_MAX_SIZE - 1) / BLOCK_MAX_SIZE;

        int threadsPerBlock = block_size;
        int blocksPerGrid = grid_size;
        // std::cout << "current hop (" << loop << "): queue_in_num (" << queue_in_num << ")\n";

        // kernel launch
        int initial_value = 0;
        // cudaMemcpyToSymbol(vf_num, &initial_value, sizeof(int));
        // cudaMemcpyToSymbol(producer_num, &initial_value, sizeof(int));
        // cudaMemcpyToSymbol(block_offset, &initial_value, sizeof(int));
        // cudaMemcpyToSymbol(processed_num, &initial_value, sizeof(int));
        bfs_hops_fusion_2<<<blocksPerGrid, threadsPerBlock>>>(d_offset, d_edges, offset.size(), edge_num,
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
    std::cout << "GPU: bfs_hops_fusion. Elapsed time :" << milliseconds << " (ms)\n";

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
