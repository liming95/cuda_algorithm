#!/bin/bash

# Check for argument
if [ -z "$1" ]; then
  echo "Usage: $0 <output_suffix>"
  exit 1
fi

# Target binary to profile
TARGET="../build/main_test"

# Output directory (relative path)
OUTPUT_DIR="../log/ncu"
mkdir -p "$OUTPUT_DIR"

# Timestamp and suffix for unique filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUFFIX="$1"
OUTPUT_NAME="${OUTPUT_DIR}/profile_${TIMESTAMP}_${SUFFIX}"

# Nsight Compute CLI tool path
NCU="/usr/local/cuda/bin/ncu"

# Step 1: Run profiler and export .ncu-rep
echo "[1/2] Running Nsight Compute profiler..."
sudo $NCU --set full --export "$OUTPUT_NAME" "$TARGET"

# Step 2: Convert .ncu-rep to .txt
echo "[2/2] Exporting .txt summary..."
sudo $NCU --import "${OUTPUT_NAME}.ncu-rep" --log-file "${OUTPUT_NAME}.txt"

# Summary
echo "✅ Done. Files saved to:"
echo "   - ${OUTPUT_NAME}.ncu-rep"
echo "   - ${OUTPUT_NAME}.txt"

