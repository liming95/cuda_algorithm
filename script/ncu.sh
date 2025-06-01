#!/bin/bash
set -e

# Check if at least two arguments are provided: output directory and target binary
if [ -z "$1" ]; then
  echo "Usage: $0 <output_dir> <target_binary> [flags...]"
  exit 1
fi

OUTPUT_DIR="$1"
TARGET="$2"
shift 2
FLAGS=("$@")

# Extract the base name of the target binary (e.g., "my_program")
BASENAME=$(basename "$TARGET")
SUFFIX="${BASENAME}"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Construct the output filename prefix
OUTPUT_NAME="${OUTPUT_DIR}/ncu_${SUFFIX}"

# Path to Nsight Compute CLI tool
NCU="/usr/local/cuda/bin/ncu"

echo "[1/2] Running Nsight Compute profiler..."
echo "Command:"
echo "sudo $NCU ${FLAGS[@]} --set full --export $OUTPUT_NAME $TARGET --profile"
# Run the profiler with the specified flags, export the results
sudo "$NCU" "${FLAGS[@]}" --set full --export "$OUTPUT_NAME" "$TARGET" --profile

# Check if the profile data file was created before exporting a text summary
if [ -f "${OUTPUT_NAME}.ncu-rep" ]; then
  echo "[2/2] Exporting .txt summary..."
  echo "Command:"
  echo "sudo $NCU --import ${OUTPUT_NAME}.ncu-rep --log-file ${OUTPUT_NAME}.txt"
  # Import the profile data and export a human-readable text report
  sudo "$NCU" --import "${OUTPUT_NAME}.ncu-rep" --log-file "${OUTPUT_NAME}.txt"
  echo "✅ Done. Files saved to:"
  echo "   - ${OUTPUT_NAME}.ncu-rep"
  echo "   - ${OUTPUT_NAME}.txt"
else
  # If the profile data file does not exist, skip the text export step
  echo "Warning: Profile data file ${OUTPUT_NAME}.ncu-rep not found, skipping text export."
fi