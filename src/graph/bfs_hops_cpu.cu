#include <iostream>
#include <queue>
#include <limits>
#include "bfs_hops.cuh"

void print_hops(int source, std::vector<int> hops) {
    std::cout << "Hops from node " << source << ":\n";
    for (size_t i = 0; i < hops.size(); ++i) {
        std::cout << "Node " << i << ": ";
        if (hops[i] == INVAILD)
            std::cout << "unreachable ";
        else
            std::cout << hops[i] << " ";
    }
    std::cout << "\n";
}

std::vector<int> computeHops(int source, const std::vector<int>& offset, const std::vector<int>& endnodes) {
    int n = offset.size() - 1;
    std::vector<int> hops(n, INVAILD);
    std::queue<int> q;

    hops[source] = 0;
    q.push(source);

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int i = offset[u]; i < offset[u + 1]; ++i) {
            int v = endnodes[i];
            if (hops[v] == INVAILD) {
                hops[v] = hops[u] + 1;
                q.push(v);
            }
        }
    }

    return hops;
}

std::vector<int> runGraphTest(std::vector<int> offset, std::vector<int> endnodes, int source) {
    auto hops = computeHops(source, offset, endnodes);

    //print_hops(source, hops);
    return hops;
}