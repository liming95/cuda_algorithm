#include <iostream>
#include <queue>
#include <limits>
#include "bfs_hops.cuh"

void print_hops(int source, std::vector<int> hops) {
    std::cout << "Hops from node " << source << ":\n";
    for (size_t i = 0; i < hops.size(); ++i) {
        std::cout << "Node " << i << ": ";
        if (hops[i] == -1)
            std::cout << "unreachable\n";
        else
            std::cout << hops[i] << "\n";
    }
}

std::vector<int> computeHops(int source, const std::vector<int>& offset, const std::vector<int>& endnodes) {
    int n = offset.size() - 1;
    std::vector<int> hops(n, -1);
    std::queue<int> q;

    hops[source] = 0;
    q.push(source);

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int i = offset[u]; i < offset[u + 1]; ++i) {
            int v = endnodes[i];
            if (hops[v] == -1) {
                hops[v] = hops[u] + 1;
                q.push(v);
            }
        }
    }

    return hops;
}

std::vector<int> runGraphTest(std::vector<int> offset, std::vector<int> endnodes, int source) {
    auto hops = computeHops(source, offset, endnodes);

    // print_hops(source, hops);
    return hops;
}