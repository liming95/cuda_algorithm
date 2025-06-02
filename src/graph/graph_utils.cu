#include "graph_utils.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>

void buildGraphCSR(std::vector<int>& offset, std::vector<int>& endnodes) {
    offset = {0, 2, 4, 5, 5, 6, 6};
    endnodes = {
        1, 2,    // node 0
        3, 4,    // node 1
        5,       // node 2
                 // node 3 has no neighbors
        5        // node 4
                 // node 5 has no neighbors
    };
}

void buildRandomGraphCSR(std::vector<int>& offset, std::vector<int>& endnodes, int num_nodes, int max_degree) {
    std::mt19937 rng(std::random_device{}()); // random generation
    std::uniform_int_distribution<int> degree_dist(0, max_degree);
    std::uniform_int_distribution<int> node_dist(0, num_nodes - 1);

    offset.clear();
    endnodes.clear();
    offset.push_back(0);

    for (int i = 0; i < num_nodes; ++i) {
        int degree = degree_dist(rng);
        std::unordered_set<int> neighbors;

        while (neighbors.size() < degree) {
            int target = node_dist(rng);
            if (target != i) {  // avoid loop
                neighbors.insert(target);
            }
        }

        for (int neighbor : neighbors) {
            endnodes.push_back(neighbor);
        }

        offset.push_back(endnodes.size());
    }
}

void printGraphCSR(const std::vector<int>& offset, const std::vector<int>& endnodes) {
    int num_nodes = offset.size() - 1;  // offset[i+1] - offset[i] gives number of neighbors of node i

    for (int i = 0; i < num_nodes; ++i) {
        std::cout << "Node " << i << " -> ";
        for (int j = offset[i]; j < offset[i + 1]; ++j) {
            std::cout << endnodes[j];
            if (j < offset[i + 1] - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
}
