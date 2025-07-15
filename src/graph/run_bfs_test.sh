#!/bin/bash
EXE_FILE=../../build/test/bin/graph/test_bfs_hops

initial_arg=50
num_runs=5
for ((i=1; i<=num_runs; i++)); do
	echo "Runing with argument: $initial_arg"
	"$EXE_FILE" "$initial_arg"
	initial_arg=$((initial_arg * 2))
done
