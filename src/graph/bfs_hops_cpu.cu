#include <iostream>
#include <queue>
#include <limits>
#include "bfs_hops.cuh"
#include "graph_utils.cuh"

std::vector<int> computeHops(int source, const std::vector<int>& offset, const std::vector<int>& endpoints) {
    int n = offset.size() - 1;
    std::vector<int> hops(n, std::numeric_limits<int>::max());
    std::queue<int> q;

    hops[source] = 0;
    q.push(source);

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int i = offset[u]; i < offset[u + 1]; ++i) {
            int v = endpoints[i];
            if (hops[v] == std::numeric_limits<int>::max()) {
                hops[v] = hops[u] + 1;
                q.push(v);
            }
        }
    }

    return hops;
}

std::vector<int> runGraphTest() {
    std::vector<int> offset, endpoints;
    buildGraphCSR(offset, endpoints);

    int source = 0;
    auto hops = computeHops(source, offset, endpoints);

    std::cout << "Hops from node " << source << ":\n";
    for (size_t i = 0; i < hops.size(); ++i) {
        std::cout << "Node " << i << ": ";
        if (hops[i] == std::numeric_limits<int>::max())
            std::cout << "unreachable\n";
        else
            std::cout << hops[i] << "\n";
    }
    return hops;
}
