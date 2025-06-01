#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <vector>

// Build a directed graph in CSR format.
// offset: start index of neighbors in the endpoints array
// endpoints: flattened list of destination nodes
void buildGraphCSR(std::vector<int>& offset, std::vector<int>& endpoints) {
    offset = {0, 2, 4, 5, 5, 6, 6};
    endpoints = {
        1, 2,    // node 0
        3, 4,    // node 1
        5,       // node 2
                 // node 3 has no neighbors
        5        // node 4
                 // node 5 has no neighbors
    };
}
#endif  // GRAPH_UTILS_H
