set(matmul_srcs
  matmul_cpu.cu
  matmul_gpu_base.cu
  matmul_tiling.cu
  matmul_prefetch.cu
  matmul_bank.cu
  matmul_mul_tiling.cu
  matmul_tiling_register.cu
)

# Metatarget for all matmuls:
add_custom_target(nvbench.matmul.all)
# add_dependencies(nvbench.all nvbench.matmul.all)


function (nvbench_add_matmuls_target target_prefix cuda_std)
  add_custom_target(${target_prefix}.all)
  add_dependencies(nvbench.matmul.all ${target_prefix}.all)

  foreach(matmul_src IN LISTS matmul_srcs)
    get_filename_component(matmul_name "${matmul_src}" NAME_WLE)
    string(PREPEND matmul_name "${target_prefix}.")
    add_executable(${matmul_name} "${matmul_src}")
    message(STATUS "Executable output dir: ${NVBench_EXECUTABLE_OUTPUT_DIR}")
    nvbench_config_target(${matmul_name})
    target_include_directories(${matmul_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    target_link_libraries(${matmul_name} PRIVATE nvbench::main)
    set_target_properties(${matmul_name} PROPERTIES COMPILE_FEATURES cuda_std_${cuda_std})
    add_test(NAME ${matmul_name}
      COMMAND "$<TARGET_FILE:${matmul_name}>" --timeout 0.1 --min-time 1e-5
    )

    # These should not deadlock. If they do, it may be that the CUDA context was created before
    # setting CUDA_MODULE_LOAD=EAGER in main, see NVIDIA/nvbench#136.
    set_tests_properties(${matmul_name} PROPERTIES
      FAIL_REGULAR_EXPRESSION "Possible Deadlock Detected"
    )

    add_dependencies(${target_prefix}.all ${matmul_name})
  endforeach()
endfunction()

message(STATUS "Adding executable for source: ${matmul_srcs}")
message(STATUS "NVBench_DETECTED_CUDA_STANDARDS = ${NVBench_DETECTED_CUDA_STANDARDS}")

foreach (std IN LISTS NVBench_DETECTED_CUDA_STANDARDS)
  nvbench_add_matmuls_target(nvbench.matmul.cpp${std} ${std})
  message(STATUS "[In processing] Adding executable for source: ${matmul_srcs}")
endforeach()
