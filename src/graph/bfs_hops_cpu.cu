#include <iostream>
#include <queue>
#include <limits>
#include <chrono>        // statistic system
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
    std::queue<int> q_in, q_out;

    hops[source] = 0;
    q_in.push(source);
    int level = 0;
    while(!q_in.empty()) {
        printf("level: %d, input size: %ld, output_size: %ld\n", level, q_in.size(), q_out.size());
        int q_size = q_in.size();
        for(int i = 0; i < q_size; i++) {
            int vertex = q_in.front(); q_in.pop();
            int start = offset[vertex];
            int end = offset[vertex+1];

            for(int j = start; j < end; j++) {
                int ngb = endnodes[j];
                if(hops[ngb] == INVAILD) {
                    hops[ngb] = hops[vertex] + 1;
                    q_out.push(ngb);
                }
            }
        }
        q_in = std::move(q_out);
        level++;
    }
    return hops;
}

std::vector<int> runGraphTest(std::vector<int> offset, std::vector<int> endnodes, int source) {
    auto hops = computeHops(source, offset, endnodes);

    //print_hops(source, hops);
    return hops;
}