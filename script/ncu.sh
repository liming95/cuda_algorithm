#!/bin/bash
set -e  # Exit on error

# Check for argument
if [ -z "$1" ]; then
  echo "Usage: $0 <binary_file_path>"
  exit 1
fi

# Get target binary from argument
TARGET="$1"

# Extract file name without path and extension
BASENAME=$(basename "$TARGET")
SUFFIX="${BASENAME%.*}"

# Output directory with current date
DATE=$(date +%F)  # Format: YYYY-MM-DD
OUTPUT_DIR="../log/ncu/${DATE}"
mkdir -p "$OUTPUT_DIR"

# Output file name
OUTPUT_NAME="${OUTPUT_DIR}/profile_${SUFFIX}"

# Nsight Compute CLI path
NCU="/usr/local/cuda/bin/ncu"

# Step 1: Run profiler and export .ncu-rep
echo "[1/2] Running Nsight Compute profiler..."
sudo "$NCU" --set full --export "$OUTPUT_NAME" "$TARGET"

# Step 2: Convert .ncu-rep to .txt
echo "[2/2] Exporting .txt summary..."
sudo "$NCU" --import "${OUTPUT_NAME}.ncu-rep" --log-file "${OUTPUT_NAME}.txt"

# Summary
echo "✅ Done. Files saved to:"
echo "   - ${OUTPUT_NAME}.ncu-rep"
echo "   - ${OUTPUT_NAME}.txt"
