# src/Makefile

TARGET = main_test
OBJS=$(BUILD_DIR)/matrix_mul.o
SRC_DIR = ../src
BUILD_DIR = ../build
LOG_DIR = ../log
NVCC = nvcc
SUB_DIRS = $(SRC_DIR)/matmul
COMPILE_FLAG = -I $(SRC_DIR)/include -gencode arch=compute_89,code=sm_89

SCRIPT_DIR = ../script
EXE_PATH = $(BUILD_DIR)/$(TARGET)
PERF_OUT_DIR = $(LOG_DIR)/perf

PERF_NAME ?=

all: build

run: build
	./$(BUILD_DIR)/$(TARGET)

perf: build
	@if [ -z "$(PERF_NAME)" ]; then \
		echo "Error: PERF_NAME is required. Please provide a value for PERF_NAME."; \
		exit 1; \
	fi
	mkdir -p $(PERF_OUT_DIR)
	@LOG_FILE="run_$(shell date +%Y%m%d_%H%M%S)_$(PERF_NAME).log" && \
	sudo $(SCRIPT_DIR)/ncu_nsys_env.sh $(EXE_PATH) $(PERF_OUT_DIR) | tee "$(PERF_OUT_DIR)/$$LOG_FILE"

build: $(SUB_DIRS) $(TARGET)

$(TARGET): $(SRC_DIR)/main.cu $(OBJS)
	@mkdir -p $(BUILD_DIR)
	nvcc $(COMPILE_FLAG) -o $(BUILD_DIR)/$@ $^

$(SUB_DIRS):
	$(MAKE) -C $@

clean:
	rm -rf $(BUILD_DIR)/main_test
	@for dir in $(SUB_DIRS); do \
		$(MAKE) -C $$dir clean; \
	done

clean_all:
	rm -rf $(BUILD_DIR)
	rm -rf $(LOG_DIR)

.PHONY: all run build clean clean_all $(SUB_DIRS)