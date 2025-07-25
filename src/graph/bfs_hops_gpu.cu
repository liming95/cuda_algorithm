#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <assert.h>
#include "bfs_hops.cuh"

template <typename T>
void print_vector (std::vector<T> vec, int num) {
    printf("[");
    if (num == -1) {
        for(int i = 0; i < vec.size(); i++){
            printf("%f ", vec[i]);
        }
    } else {
        auto n = num < vec.size() ? num : vec.size();
        for(int i = 0; i < n; i++) {
            printf("%f ", vec[i]);
        }
    }
    printf("]\n");

}
__global__ void empty_time(){
}
__global__ void computeHops_gpu_noTimer(int* queue_in, int* queue_out, int* offset, int* endnodes,
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
            //printf("edge:(%d, %d)  ", node, end_node);
            // calculate hops from start to end_node
            end_hop = hops[node] + 1;
            end_hop = atomicCAS(&hops[end_node], INVAILD, end_hop);

            // push into queue_out
            if(end_hop == INVAILD) {
                queue_index = atomicAdd(queue_out_num, 1);
                queue_out[queue_index] = end_node;
                // if(end_node >= 32 * 256)
                //     printf("end_node: %d\n", end_node);
            }
        }
        //printf("\n");
    }
    // if(tid == 0){
    //     printf("queue_out_num: %d\n", *queue_out_num);
    // }
}

struct Timer_B {
    unsigned long long total;
};

struct Timer_W {
    unsigned long long total;
    unsigned long long get_input;
    unsigned long long get_offset;
    unsigned long long rmd;                // traverse, update, write back
};
struct Timer_T {
    unsigned long long total;
    unsigned long long traverse;
    unsigned long long update_get;
    unsigned long long update_atomic;
    unsigned long long write_atomic;
    unsigned long long write_back;
};

struct Timer {
    struct Timer_B blk_t;
    struct Timer_W warp_t[BLOCK_MAX_SIZE / 32];
    struct Timer_T thread_t[BLOCK_MAX_SIZE];
};

void print_timer(struct Timer timer, int threadsPerBlock) {
    int clock_rate_khz;
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
    float clock_ghz = clock_rate_khz / 1000000.0f;
    printf("-----------------------------PRINT START--------------------------------\n");

    printf("clock frequency: %.2f GHz\n", clock_ghz);
    printf("block[0] elapsed time: %llu (%.2f ns)\n", timer.blk_t.total, timer.blk_t.total / clock_ghz);
    int warp_num = threadsPerBlock / 32;
    for (int i = 0; i < warp_num; i++) {
        printf("warp[%d] elapsed time:\n", i);
        if(timer.warp_t[i].total == 0) continue;
        printf("  total: %llu (%.2f ns)", timer.warp_t[i].total, timer.warp_t[i].total / clock_ghz);
        printf("  get_input: %llu (%.2f ns)", timer.warp_t[i].get_input, timer.warp_t[i].get_input / clock_ghz);
        printf("  get_offset: %llu (%.2f ns)", timer.warp_t[i].get_offset, timer.warp_t[i].get_offset / clock_ghz);
        printf("  rmd: %llu (%.2f ns)\n", timer.warp_t[i].rmd, timer.warp_t[i].rmd / clock_ghz);
    }

    for (int i = 0; i < threadsPerBlock; i++){
        if(timer.thread_t[i].total == 0) continue;
        printf("thread[%d] elased time:\n", i);

        printf(" total: %llu (%.2f ns)", timer.thread_t[i].total, timer.thread_t[i].total / clock_ghz);
        printf(" traverse: %llu (%.2f ns)", timer.thread_t[i].traverse, timer.thread_t[i].traverse / clock_ghz);
        printf(" update_get: %llu (%.2f ns)", timer.thread_t[i].update_get, timer.thread_t[i].update_get / clock_ghz);
        printf(" update_atomic: %llu (%.2f ns)", timer.thread_t[i].update_atomic, timer.thread_t[i].update_atomic / clock_ghz);
        printf(" write_atomic: %llu (%.2f ns)", timer.thread_t[i].write_atomic, timer.thread_t[i].write_atomic / clock_ghz);
        printf(" write_back: %llu (%.2f ns)\n", timer.thread_t[i].write_back, timer.thread_t[i].write_back / clock_ghz);
    }
    printf("-----------------------------PRINT END--------------------------------\n\n");
}

void print_average_csv(struct Timer timer, int threadsPerBlock){
    int clock_rate_khz;
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
    float clock_ghz = clock_rate_khz / 1000000.0f;
    printf("-----------------------------PRINT START--------------------------------\n");

    printf("clock frequency: %.2f GHz\n", clock_ghz);
    printf("block[0] elapsed time: %llu (%.2f ns)\n", timer.blk_t.total, timer.blk_t.total / clock_ghz);
    int warp_num = threadsPerBlock / 32;
    unsigned long long total = 0, get_input = 0, get_offset = 0, rmd = 0;
    printf("warp elapsed time:\n");
    printf("total get_input get_offset rmd\n");
    int warp_num_used = 0;
    for (int i = 0; i < warp_num; i++) {

        if(timer.warp_t[i].total == 0) continue;
        warp_num_used++;
        total += timer.warp_t[i].total;
        get_input += timer.warp_t[i].get_input;
        get_offset += timer.warp_t[i].get_offset;
        rmd += timer.warp_t[i].rmd;
    }
    assert(warp_num_used != 0);
    printf("cycle: %llu %llu %llu %llu\n", total/warp_num_used, get_input/warp_num_used, get_offset/warp_num_used, rmd/warp_num_used);

    unsigned long long traverse = 0, update_get = 0, update_atomic = 0, write_atomic = 0, write_back = 0;
    printf("thread elased time:\n");
    printf("total traverse update_get update_atomic write_atomic write_back\n");
    int thread_num_used = 0;
    for (int i = 0; i < threadsPerBlock; i++){
        if(timer.thread_t[i].total == 0) continue;
        thread_num_used++;
        total += timer.thread_t[i].total;
        traverse += timer.thread_t[i].traverse;
        update_get += timer.thread_t[i].update_get;
        update_atomic += timer.thread_t[i].update_atomic;
        write_atomic += timer.thread_t[i].write_atomic;
        write_back += timer.thread_t[i].write_back;
    }
    printf("cycle: %llu %llu %llu %llu %llu %llu\n", total/thread_num_used, traverse/thread_num_used,
                update_get/thread_num_used, update_atomic/thread_num_used,
                write_atomic/thread_num_used, write_back/thread_num_used);
    printf("-----------------------------PRINT END--------------------------------\n\n");
}

__global__ void computeHops_gpu(int* queue_in, int* queue_out, int* offset, int* endnodes,
                                int* hops, int queue_in_num, int* queue_out_num, int node_num,
                                int edge_num, struct Timer * timer) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int node, end_node;
    unsigned long long blk_clk_start, blk_clk_end;
    unsigned long long warp_clk_start, warp_clk_in, warp_clk_offset, warp_clk_end;
    unsigned long long thd_clk_start, thd_clk_traverse, thd_clk_update1, thd_clk_update2, thd_clk_out_start, thd_clk_out1, thd_clk_out2, thd_clk_end;
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) blk_clk_start = clock64();

    if (tid < queue_in_num) {
        //__syncwarp();
        warp_clk_start = clock64();
        node = queue_in[tid];
        warp_clk_in = clock64();

        int edge_start = offset[node];
        int edge_end = offset[node+1];
        warp_clk_offset = clock64();
        if(blockIdx.x == 0) {
            (timer+blockIdx.x)->thread_t[threadIdx.x].total = 0;
            (timer+blockIdx.x)->thread_t[threadIdx.x].traverse = 0;
            (timer+blockIdx.x)->thread_t[threadIdx.x].update_get = 0;
            (timer+blockIdx.x)->thread_t[threadIdx.x].update_atomic = 0;
            (timer+blockIdx.x)->thread_t[threadIdx.x].write_atomic = 0;
            (timer+blockIdx.x)->thread_t[threadIdx.x].write_back = 0;
        }

        int end_hop, queue_index;
        for(int i = edge_start; i < edge_end; i++) {
            thd_clk_start = clock64();
            end_node = endnodes[i];
            thd_clk_traverse = clock64();

            // calculate hops from start to end_node
            end_hop = hops[node] + 1;
            thd_clk_update1 = clock64();
            end_hop = atomicCAS(&hops[end_node], INVAILD, end_hop);
            thd_clk_update2 = clock64();

            // push into queue_out
            if(end_hop == INVAILD) {
                thd_clk_out_start = clock64();
                queue_index = atomicAdd(queue_out_num, 1);
                thd_clk_out1 = clock64();
                queue_out[queue_index] = end_node;
                thd_clk_out2 = clock64();
                if(blockIdx.x == 0) {
                    (timer+blockIdx.x)->thread_t[threadIdx.x].write_atomic += thd_clk_out1 - thd_clk_out_start;
                    (timer+blockIdx.x)->thread_t[threadIdx.x].write_back += thd_clk_out2 - thd_clk_out1;
                }
            }
            thd_clk_end = clock64();
            if(blockIdx.x == 0) {
                (timer+blockIdx.x)->thread_t[threadIdx.x].total += thd_clk_end - thd_clk_start;
                (timer+blockIdx.x)->thread_t[threadIdx.x].traverse += thd_clk_traverse - thd_clk_start;
                (timer+blockIdx.x)->thread_t[threadIdx.x].update_get += thd_clk_update1 - thd_clk_traverse;
                (timer+blockIdx.x)->thread_t[threadIdx.x].update_atomic += thd_clk_update2 - thd_clk_update1;
            }
        }
        //__syncwarp();
        warp_clk_end = clock64();

        if(threadIdx.x % 32 == 0 && blockIdx.x == 0) {
            int warp_no = threadIdx.x / 32;
            (timer+blockIdx.x)->warp_t[warp_no].total = warp_clk_end - warp_clk_start;
            (timer+blockIdx.x)->warp_t[warp_no].get_input = warp_clk_in - warp_clk_start;
            (timer+blockIdx.x)->warp_t[warp_no].get_offset = warp_clk_offset - warp_clk_in;
            (timer+blockIdx.x)->warp_t[warp_no].rmd = warp_clk_end - warp_clk_offset;
        }
    }
     __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        blk_clk_end = clock64();
        (timer+blockIdx.x)->blk_t.total = blk_clk_end - blk_clk_start;
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
    struct Timer timer;
    struct Timer * d_timer;

    cudaMalloc(&d_offset, offset_size);
    cudaMalloc(&d_endnodes, endnodes_size);
    cudaMalloc(&d_queue_in, queue_size);
    cudaMalloc(&d_queue_out, queue_size);
    cudaMalloc(&d_hops, hops_size);
    //cudaMalloc(&d_queue_in_num, queue_num_size);
    cudaMalloc(&d_queue_out_num, queue_num_size);
    cudaMalloc(&d_timer, sizeof(timer));

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

    // the execution time  of each kernel;
    cudaEvent_t initial_start, initial_stop, compute_start, compute_stop, mem_start, mem_stop;
    cudaEventCreate(&initial_start);
    cudaEventCreate(&initial_stop);
    cudaEventCreate(&compute_start);
    cudaEventCreate(&compute_stop);
    cudaEventCreate(&mem_start);
    cudaEventCreate(&mem_stop);

    std::vector<float> initial_time, compute_time, mem_time;

    auto cpu_start_time = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start, 0);
    int sum = 0, level = 0;
    do {
        sum += queue_in_num;
        // block size
        int threadsPerBlock_up = ((queue_in_num + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        int block_size = queue_in_num > BLOCK_MAX_SIZE ? BLOCK_MAX_SIZE : threadsPerBlock_up;
        int grid_size = (queue_in_num + BLOCK_MAX_SIZE - 1) / BLOCK_MAX_SIZE;

        int threadsPerBlock = block_size;
        int blocksPerGrid = grid_size;
        // std::cout << "current hop (" << loop << "): queue_in_num (" << queue_in_num << ")\n";

        // kernel launch
        cudaEventRecord(compute_start, 0);
        computeHops_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_queue_in, d_queue_out, d_offset, d_endnodes,
                                                            d_hops, queue_in_num, d_queue_out_num, offset.size(),
                                                            edge_num, d_timer);
        cudaEventRecord(compute_stop, 0);
        cudaEventSynchronize(compute_stop);
        float compute_time_each = 0;
        cudaEventElapsedTime(&compute_time_each, compute_start, compute_stop);
        compute_time.push_back(compute_time_each);

        CHECK_CUDA_SYNC("After device synchronize");

        // copy queue_out_num to host
        cudaEventRecord(mem_start, 0);
        cudaMemcpy(&queue_out_num, d_queue_out_num, queue_num_size, cudaMemcpyDeviceToHost);
        cudaEventRecord(mem_stop, 0);
        cudaEventSynchronize(mem_stop);
        float mem_time_each = 0;
        cudaEventElapsedTime(&mem_time_each, mem_start, mem_stop);
        mem_time.push_back(mem_time_each);

        queue_in_num = queue_out_num;
        level++;

        //printf("queue_out_num: %d\n", queue_out_num);
        queue_out_num = 0;

        cudaEventRecord(initial_start, 0);
        cudaMemcpy(d_queue_out_num, &queue_out_num, queue_num_size, cudaMemcpyHostToDevice);
        cudaEventRecord(initial_stop, 0);
        cudaEventSynchronize(initial_stop);
        float initial_time_each = 0;
        cudaEventElapsedTime(&initial_time_each, initial_start, initial_stop);
        initial_time.push_back(initial_time_each);

        //Timer
        printf("level: %d, vertex num: %d\n", level, queue_in_num);
        cudaMemcpy(&timer, d_timer, sizeof(timer), cudaMemcpyDeviceToHost);
        print_average_csv(timer, threadsPerBlock);
        int * tmp;
        tmp = d_queue_in;
        d_queue_in = d_queue_out;
        d_queue_out = tmp;
    }
    while(queue_in_num);
    //printf("gpu sum: %d\n", sum);
    // copy hops from device to host
    cudaMemcpy(hops.data(), d_hops, hops_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    float cpu_launch_time = std::chrono::duration<float, std::milli>(cpu_end_time - cpu_start_time).count();

    float gpu_launch_time = 0;
    cudaEventElapsedTime(&gpu_launch_time, start, stop);

    float total_initial_time = 0, total_compute_time = 0, total_mem_time = 0;
    for(int i = 0; i < compute_time.size(); i++){
        total_initial_time += initial_time[i];
        total_compute_time += compute_time[i];
        total_mem_time += mem_time[i];
    }
    std::vector<float> kernel_launch_time;
    float total_launch_time = 0;
    for (int i = 0; i < level; i++) {
        cudaEventRecord(start, 0);
        empty_time<<<1, 1>>>();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float launch_time_each = 0;
        cudaEventElapsedTime(&launch_time_each, initial_start, initial_stop);
        initial_time.push_back(launch_time_each);
        total_launch_time += launch_time_each;
    }

    std::cout << "bfs_hops_gpu START\n";
    std::cout << "CPU: total time: " << cpu_launch_time << " (ms)\n";
    std::cout << "GPU: total elapsed time :" << gpu_launch_time << " (ms)\n";
    std::cout << "stage1 (initial) time :" << total_initial_time << " (ms)\n";
    print_vector<float>(initial_time, -1);
    std::cout << "stage2: (compute) time :" << total_compute_time << " (ms)\n";
    print_vector<float>(compute_time, -1);
    std::cout << "stage3: (mem) time :" << total_mem_time << " (ms)\n";
    print_vector<float>(mem_time, -1);
    std::cout << "launch time: " << total_launch_time / level << "(ms)\n";
    std::cout << "bfs_hops_gpu END\n";


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
