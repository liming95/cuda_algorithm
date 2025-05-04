# src/matmul/Makefile

BUILD_DIR = ../../build
COMPILER = nvcc
SRC_DIR = ../../src/matmul
COMPILE_FLAG = -I ../include

# Find all .cu files in the source directory
SRCS = $(wildcard $(SRC_DIR)/*.cu)

# Generate object files list
OBJS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SRCS))

all: $(OBJS)
# Compile .cu files to .o files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	$(COMPILER) $(COMPILE_FLAG) -c $< -o $@

# Clean up object files
clean:
	rm -f $(OBJS)

.PHONY: all clean