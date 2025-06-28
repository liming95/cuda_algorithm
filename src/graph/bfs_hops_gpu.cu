#include <iostream>
#include <cuda_runtime.h>
#include "bfs_hops.cuh"


__global__ void computeHops_gpu(int* queue_in, int* queue_out, int* offset, int* endnodes,
                                int* hops, int queue_in_num, int* queue_out_num, int node_num,
                                int edge_num) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int node, end_node;
    if (tid < queue_in_num) {
        node = queue_in[tid];
        int edge_start = offset[node];
        int edge_end = offset[node+1];

        int end_hop, queue_index;
        for(int i = edge_start; i < edge_end; i++) {
            end_node = endnodes[i];

            // calculate hops from start to end_node
            end_hop = hops[node] + 1;
            end_hop = atomicCAS(&hops[end_node], INVAILD, end_hop);

            // push into queue_out
            if(end_hop == INVAILD) {
                queue_index = atomicAdd(queue_out_num, 1);
                queue_out[queue_index] = end_node;
            }
        }
    }
}

std::vector<int> test_bfs_hops_gpu(std::vector<int> offset, std::vector<int> endnodes, int source){
    // build queue
    int * queue_in;
    // * queue_out;
    int node_num = offset.size() - 1;
    int edge_num = endnodes.size();
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
    int endnodes_size = edge_num * sizeof(int);
    int queue_size = node_num * sizeof(int);
    int hops_size = offset_size;
    int * d_offset, *d_endnodes, *d_queue_in, *d_queue_out, *d_hops, *d_queue_out_num;

    cudaMalloc(&d_offset, offset_size);
    cudaMalloc(&d_endnodes, endnodes_size);
    cudaMalloc(&d_queue_in, queue_size);
    cudaMalloc(&d_queue_out, queue_size);
    cudaMalloc(&d_hops, hops_size);
    //cudaMalloc(&d_queue_in_num, queue_num_size);
    cudaMalloc(&d_queue_out_num, queue_num_size);

    cudaMemcpy(d_offset, offset.data(), offset_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_endnodes, endnodes.data(), endnodes_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_queue_in, queue_in, queue_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hops, hops.data(), hops_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_queue_out_num, &queue_out_num, queue_num_size, cudaMemcpyHostToDevice);
    // std::cout << "GPU no perf\n";
    // int loop = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    do {
        // block size
        int threadsPerBlock_up = ((queue_in_num + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        int block_size = queue_in_num > BLOCK_MAX_SIZE ? BLOCK_MAX_SIZE : threadsPerBlock_up;
        int grid_size = (queue_in_num + BLOCK_MAX_SIZE - 1) / BLOCK_MAX_SIZE;

        int threadsPerBlock = block_size;
        int blocksPerGrid = grid_size;
        // std::cout << "current hop (" << loop << "): queue_in_num (" << queue_in_num << ")\n";

        // kernel launch
        computeHops_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_queue_in, d_queue_out, d_offset, d_endnodes,
                                                            d_hops, queue_in_num, d_queue_out_num, offset.size(),
                                                            edge_num);

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
    }
    while(queue_in_num);

    // copy hops from device to host
    cudaMemcpy(hops.data(), d_hops, hops_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU: bfs_hops_gpu. Elapsed time :" << milliseconds << " (ms)\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_offset);
    cudaFree(d_endnodes);
    cudaFree(d_queue_in);
    cudaFree(d_queue_out);
    cudaFree(d_hops);
    cudaFree(d_queue_out_num);

    //print_hops(source, hops);
    return hops;
}
