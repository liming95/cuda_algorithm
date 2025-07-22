# src/graph

# The name of the final executable
TARGET_OBJ = test_bfs_hops

# Source and build directories
SRC_DIR = ./
BUILD_DIR = ../../build/test/bin/graph

# Compiler and compilation flags
COMPILER = nvcc
COMPILER_FLAGS = -gencode arch=compute_89,code=sm_89 -std=c++17 -O2

# All source files required for compilation
SRC_FILES = $(SRC_DIR)/main.cu $(SRC_DIR)/bfs_hops_cpu.cu $(SRC_DIR)/bfs_hops_gpu.cu \
			./graph_utils.cu ./bfs_hops_async.cu ./bfs_hops_fusion.cu ./bfs_hops_async_update.cu \
			./bfs_hops_fusion_o1.cu ./bfs_hops_async_update_o1.cu ./bfs_hops_fusion_o2.cu ./bfs_hops_async_update_o2.cu

PERF_TOOL = ../../script/ncu.sh
PERF_OUT_DIR = ../../build/test/perf/graph
RUN_ARGS = 40000
# Default target
all: build

# Build target: compiles the program
build: $(BUILD_DIR)/$(TARGET_OBJ)

# Rule to compile the final executable from source files
$(BUILD_DIR)/$(TARGET_OBJ): $(SRC_FILES)
	@mkdir -p $(BUILD_DIR)
	$(COMPILER) $(COMPILER_FLAGS) -o $@ $^
	@echo "$(BUILD_DIR)/$(TARGET_OBJ)"

# Run the compiled program
run: build
	$(BUILD_DIR)/$(TARGET_OBJ) $(RUN_ARGS)

# Clean up generated build files
clean:
	rm -f $(BUILD_DIR)/$(TARGET_OBJ)

perf: build
	@mkdir -p $(PERF_OUT_DIR)
	$(PERF_TOOL) $(PERF_OUT_DIR) $(BUILD_DIR)/$(TARGET_OBJ)

# Declare phony targets that are not actual files
.PHONY: all run build clean perf