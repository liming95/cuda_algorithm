# src/Makefile

TARGET = main_test
SRC_DIR = ../src
BUILD_DIR = ../build
NVCC = nvcc
SUB_DIRS = $(SRC_DIR)/matmul
COMPILE_FLAG = -I $(SRC_DIR)/include

SCRIPT_DIR = ../script
EXE_PATH = $(BUILD_DIR)/$(TARGET)
PERF_OUT_DIR = $(BUILD_DIR)/perf

all: build

run: build
	./$(BUILD_DIR)/$(TARGET)

perf: build
	mkdir -p $(PERF_OUT_DIR)
	sudo $(SCRIPT_DIR)/ncu_nsys_env.sh $(EXE_PATH) $(PERF_OUT_DIR)

build: $(SUB_DIRS) $(TARGET)

$(TARGET): $(SRC_DIR)/main.cu
	@mkdir -p $(BUILD_DIR)
	@OBJS=$(wildcard $(BUILD_DIR)/*.o) && \
	nvcc $(COMPILE_FLAG) -o $(BUILD_DIR)/$@ $^ $$OBJS

$(SUB_DIRS):
	$(MAKE) -C $@

clean:
	rm -rf $(BUILD_DIR)/main_test
	@for dir in $(SUB_DIRS); do \
		$(MAKE) -C $$dir clean; \
	done

clean_all:
	rm -rf $(BUILD_DIR)

.PHONY: all run build clean $(SUB_DIRS)