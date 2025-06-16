#include <assert.h>
#include "bfs_hops.cuh"
#include "graph_utils.cuh"

int main() {
    int source = 0;
    std::vector<int> offset, endnodes;
    buildGraphCSR(offset, endnodes);
    int node_num = 40;
    int max_degree = 10;
    //buildRandomGraphCSR(offset, endnodes, node_num, max_degree);
    //printGraphCSR(offset, endnodes);
    std::vector<int> hops_cpu, hops_gpu, hops_async;


    hops_cpu = runGraphTest(offset, endnodes, source);
    //hops_gpu = test_bfs_hops_gpu(offset, endnodes, source);
    //assert(hops_cpu == hops_gpu);
    hops_async = test_bfs_hops_async(offset, endnodes, source);
    //assert(hops_cpu == hops_gpu);
    return 0;
}
