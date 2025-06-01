#include <assert.h>
#include "bfs_hops.cuh"

int main() {
    std::vector<int> hops_cpu, hops_gpu;
    hops_cpu = runGraphTest();
    hops_gpu = test_bfs_hops_gpu();
    assert(hops_cpu == hops_gpu);
    return 0;
}
