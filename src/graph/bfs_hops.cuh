#ifndef GRAPH_ALGORITHMS_H
#define GRAPH_ALGORITHMS_H

#include <vector>

#define WARP_SIZE 32
#define BLOCK_MAX_SIZE 256
#define CHECK_CUDA_SYNC(msg)                                            \
    {                                                                   \
        cudaError_t err = cudaDeviceSynchronize();                      \
        if (err != cudaSuccess) {                                       \
            std::cerr << "[CUDA SYNC ERROR] " << msg << ": "            \
                      << cudaGetErrorString(err) << " ("                \
                      << err << ")"                                     \
                      << " at " << __FILE__ << ":" << __LINE__         \
                      << std::endl;                                     \
        }                                                               \
    }

void print_hops(int source, std::vector<int> hops);

// Compute number of hops from source to all other nodes.
std::vector<int> computeHops(int source, const std::vector<int>& offset, const std::vector<int>& endnodes);

// Run a test on the graph algorithms
std::vector<int> runGraphTest(std::vector<int> offset, std::vector<int> endnodes, int source);

std::vector<int> test_bfs_hops_gpu(std::vector<int> offset, std::vector<int> endnodes, int source);


#endif // GRAPH_ALGORITHMS_H
