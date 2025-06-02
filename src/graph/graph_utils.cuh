#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <vector>

// Build a directed graph in CSR format.
// offset: start index of neighbors in the endnodes array
// endnodes: flattened list of destination nodes
void buildGraphCSR(std::vector<int>& offset, std::vector<int>& endnodes);
void buildRandomGraphCSR(std::vector<int>& offset, std::vector<int>& endnodes, int num_nodes, int max_degree);
void printGraphCSR(const std::vector<int>& offset, const std::vector<int>& endnodes);
#endif  // GRAPH_UTILS_H
