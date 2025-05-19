#include <iostream>
#include <nvbench/nvbench.cuh>
#include <cuda/std/chrono>
#include <cuda_runtime.h>
#include "matmul.cuh"

using namespace std;

void matmul_cpu(float* A, float* B, float* C, int N){
    for(int i = 0; i < N; i++){
	    for(int j = 0; j < N; j++){
	        C[i*N+j] = 0.0f;
	        for(int k = 0; k < N; k++){
	            C[i*N+j] += A[i*N+k] * B[k*N+j];
	        }
	    }
    }
}

void matmul_cpu_benchmark(nvbench::state &state)
{
  const auto matrix_dim = static_cast<int>(state.get_int64("MatrixDim"));

  state.exec(nvbench::exec_tag::timer, [matrix_dim](nvbench::launch &, auto &timer) {
    // initial matrix
    int matrix_size = matrix_dim * matrix_dim;
    float * mat1 = new float[matrix_size];
    float * mat2 = new float[matrix_size];
    float * result = new float[matrix_size];
    fill_arry(mat1, mat2, matrix_dim);

    // Do any setup work before starting the timer here...
    timer.start();
    matmul_cpu(mat1, mat2, result, matrix_dim);
    timer.stop();
    // Any per-run cleanup here...

    //check result

    delete [] mat1;
    delete [] mat2;
    delete [] result;
  });
}
NVBENCH_BENCH(matmul_cpu_benchmark)
  .add_int64_power_of_two_axis("MatrixDim", nvbench::range(5, 10, 1))
  .set_is_cpu_only(true);