#!/bin/bash

# Get executable path and output directory from command-line arguments
EXE_DIR="$1"       # Argument 1: path to the executable (e.g., ../build/main_test)
PERF_DIR="$2"      # Argument 2: directory to save profiling reports (e.g., ../build)

# Check if arguments are provided
if [[ -z "$EXE_DIR" || -z "$PERF_DIR" ]]; then
    echo "Usage: $0 <exe_path> <perf_output_dir>"
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$PERF_DIR"

# Generate timestamp for unique output names
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_NAME="main_test_${TIMESTAMP}"

echo "Profiling results will be saved as: $BASE_NAME"

# Run NVIDIA Compute Utility (NCU)
echo "========== Start NVIDIA Compute Utility (NCU) Analysis =========="
PATH=/usr/local/cuda/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
ncu --cache-control none --clock-control none --metrics gpu__time_duration.sum "$EXE_DIR"
echo "========== End NCU Analysis =========="

# Run Nsight Systems profiler (NSYS)
echo "========== Start Nsight Systems (NSYS) Profiling =========="
nsys profile -o "$PERF_DIR/$BASE_NAME" "$EXE_DIR"
echo "========== End NSYS Profiling =========="

# Generate summary stats from the NSYS report
echo "========== Start NSYS Stats Report =========="
nsys stats "$PERF_DIR/$BASE_NAME.nsys-rep"
echo "========== End NSYS Stats Report =========="
