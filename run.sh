#!/bin/bash

# Set build and output directories
BUILD_DIR=./build
EXE_DIR="$BUILD_DIR/bin"
LOG_DIR=./log/report
REPORT_DIR="$LOG_DIR/$(date +%Y%m%d)"

# Show help message if no argument is provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 [build|exec]"
  echo "  build     Configure and build the project"
  echo "  exec      Run executables in $EXE_DIR and log output to $REPORT_DIR"
  exit 1
fi

# If argument is "build", clean and build the project
if [ "$1" == "build" ]; then
  if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"/*
  else
    mkdir -p "$BUILD_DIR"
  fi

  cd "$BUILD_DIR"
  cmake -DCMAKE_CUDA_ARCHITECTURES=native ..
  make
  cd ..
fi

# If argument is "exec", run all executables and save logs
if [ "$1" == "exec" ]; then
  mkdir -p "$REPORT_DIR"
  echo "Running executables in $EXE_DIR"

  for exe_file in "$EXE_DIR"/*; do
    if [ -x "$exe_file" ]; then
      exe_name=$(basename "$exe_file")
      echo "Running $exe_name"
      "$exe_file" | tee "$REPORT_DIR/${exe_name}.log"
    fi
  done
fi
