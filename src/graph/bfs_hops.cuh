#ifndef GRAPH_ALGORITHMS_H
#define GRAPH_ALGORITHMS_H

#include <vector>

#define WARP_SIZE 32
#define BLOCK_MAX_SIZE 256

// Compute number of hops from source to all other nodes.
std::vector<int> computeHops(int source, const std::vector<int>& offset, const std::vector<int>& endpoints);

// Run a test on the graph algorithms
std::vector<int> runGraphTest();

std::vector<int> test_bfs_hops_gpu();


#endif // GRAPH_ALGORITHMS_H
