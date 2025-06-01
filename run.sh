#!/bin/bash
set -euo pipefail

# Set directories
BUILD_DIR=./build
EXE_DIR="$BUILD_DIR/bin"
LOG_DIR=./log/report
REPORT_DIR="$LOG_DIR/$(date +%Y%m%d)"
PERF_LOG_DIR=./log/ncu
PROFILE_TOOL=./script/ncu.sh
PERF_FLAG=()
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
PERF_DIR="${PERF_LOG_DIR}/${TIMESTAMP}"

usage() {
  echo "Usage: $0 [build|exec|perf] [keyword]"
  echo "  build           Configure and build the project"
  echo "  exec [keyword]  Run executables whose names contain [keyword]"
  echo "  perf [keyword]  Profile executables whose names contain [keyword]"
  exit 1
}

if [ $# -lt 1 ]; then
  usage
fi

MODE="$1"
FILTER_KEYWORD="${2:-}"

if [ "$MODE" == "build" ]; then
  if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"/*
  else
    mkdir -p "$BUILD_DIR"
  fi

  cd "$BUILD_DIR"
  cmake -DCMAKE_CUDA_ARCHITECTURES=native ..
  make
  cd ..

elif [ "$MODE" == "exec" ]; then
  if [ ! -d "$EXE_DIR" ]; then
    echo "Executable directory $EXE_DIR does not exist."
    exit 1
  fi

  mkdir -p "$REPORT_DIR"
  echo "Running executables in $EXE_DIR matching *$FILTER_KEYWORD*"

  for exe_file in "$EXE_DIR"/*; do
    if [ -x "$exe_file" ]; then
      exe_name=$(basename "$exe_file")
      if [[ -z "$FILTER_KEYWORD" || "$exe_name" == *"$FILTER_KEYWORD"* ]]; then
        echo "Running $exe_name"
        "$exe_file" | tee "$REPORT_DIR/${exe_name}.log"
      fi
    fi
  done

elif [ "$MODE" == "perf" ]; then
  if [ ! -d "$EXE_DIR" ]; then
    echo "Executable directory $EXE_DIR does not exist."
    exit 1
  fi

  if [ ! -x "$PROFILE_TOOL" ]; then
    echo "Profile tool $PROFILE_TOOL not found or not executable."
    exit 1
  fi

  mkdir -p "$PERF_DIR"
  echo "Profiling executables in $EXE_DIR matching *$FILTER_KEYWORD*"

  for exe_file in "$EXE_DIR"/*; do
    if [ -x "$exe_file" ]; then
      exe_name=$(basename "$exe_file")
      if [[ -z "$FILTER_KEYWORD" || "$exe_name" == *"$FILTER_KEYWORD"* ]]; then
        echo "Profiling $exe_name"
        "$PROFILE_TOOL" "$PERF_DIR" "$exe_file" "${PERF_FLAG[@]}"
      fi
    fi
  done

else
  echo "Unknown mode: $MODE"
  usage
fi