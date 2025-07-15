#include <assert.h>
#include "bfs_hops.cuh"
#include "graph_utils.cuh"
#include <iostream>
void print_vector_diff(int source, std::vector<int> hops1, std::vector<int> hops2) {
    std::cout << "Hops from node " << source << ":\n";
    for (size_t i = 0; i < hops1.size(); ++i) {
        if (hops1[i] != hops2[i]) {
            std::cout << "Node " << i << ": ";
            std::cout << "(" << hops1[i] << ", " << hops2[i] << ")\n";
        }
    }
    std::cout.flush();
}
int main(int argc, char* argv[]) {
    int source = 0;
    std::vector<int> offset, edges;
    //buildGraphCSR(offset, edges);
    int node_num = std::stoi(argv[1]);
    int max_degree = 10;
    buildRandomGraphCSR(offset, edges, node_num, max_degree);
    //printGraphCSR(offset, edges);
    std::vector<int> hops_cpu, hops_gpu, hops_async, hops_fusion, hops_async_2, hops_fusion_o1, hops_async_o1;


    hops_cpu = runGraphTest(offset, edges, source);
    hops_gpu = test_bfs_hops_gpu(offset, edges, source);
    assert(hops_cpu == hops_gpu);
    hops_async = test_bfs_hops_async(offset, edges, source);
    assert(hops_cpu == hops_async);
    hops_fusion = test_bfs_hops_fusion(offset, edges, source);
    assert(hops_cpu == hops_fusion);
    hops_async_2 = test_bfs_hops_async_2(offset, edges, source);
    assert(hops_cpu == hops_async_2);
    hops_async_o1 = test_bfs_hops_async_o1(offset, edges, source);
    assert(hops_cpu == hops_async_o1);

    hops_fusion_o1 = test_bfs_hops_fusion_o1(offset, edges, source);
    print_vector_diff(source, hops_cpu, hops_fusion_o1);
    assert(hops_cpu == hops_fusion_o1);

    return 0;
}
