#ifndef GRAPH_ALGORITHMS_H
#define GRAPH_ALGORITHMS_H

#include <vector>
#include <climits>

#define INT_BIT_LEN 32
#define EVEN 0
#define ODD 1
#define INVAILD INT_MAX

#define BYTE_SIZE 8
#define LANE_NUM 3
#define WARP_SIZE 32
#define BLOCK_MAX_SIZE 256

//#define DEBUG_LEVEL 0 //Todo: 0 no debug, 1 debug, 2 log
#if DEBUG_LEVEL >= 1
    #define DEBUG_PRINT(fmt, ...) printf("[DEBUG] " fmt, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...)
#endif


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

#define GET_BIT_INDEX_OFFSET(pos, index, offset) \
    int word_bit_len = (sizeof(int)) * BYTE_SIZE; \
    index = pos / word_bit_len; \
    offset = pos % word_bit_len

void print_hops(int source, std::vector<int> hops);

// Compute number of hops from source to all other nodes.
std::vector<int> computeHops(int source, const std::vector<int>& offset, const std::vector<int>& endnodes);

// Run a test on the graph algorithms
std::vector<int> runGraphTest(std::vector<int> offset, std::vector<int> endnodes, int source);

std::vector<int> test_bfs_hops_gpu(std::vector<int> offset, std::vector<int> endnodes, int source);

std::vector<int> test_bfs_hops_async(std::vector<int> offset, std::vector<int> endnodes, int source);

std::vector<int> test_bfs_hops_fusion(std::vector<int> offset, std::vector<int> endnodes, int source);

std::vector<int> test_bfs_hops_async_2(std::vector<int> offset, std::vector<int> endnodes, int source);

std::vector<int> test_bfs_hops_fusion_o1(std::vector<int> offset, std::vector<int> endnodes, int source);

#endif // GRAPH_ALGORITHMS_H
