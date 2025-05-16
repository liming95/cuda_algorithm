# src/matmul/Makefile

BUILD_DIR = ../../build
COMPILER = nvcc
SRC_DIR = ../../src/matmul
INCLUDE_DIR = ../include
COMPILE_FLAG = -I ../include -gencode arch=compute_89,code=sm_89

# Find all .cu files in the source directory
SRCS = $(wildcard $(SRC_DIR)/*.cu)

# Generate object files list
OBJS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SRCS))

all: $(OBJS)
# Compile .cu files to .o files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(INCLUDE_DIR)/matrix_mul.h
	@mkdir -p $(BUILD_DIR)
	$(COMPILER) $(COMPILE_FLAG) -c $< -o $@

# Clean up object files
clean:
	rm -f $(OBJS)

.PHONY: all clean