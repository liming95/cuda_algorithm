//matrix_mul.h

#define MATRIX_SIZE 512
#define TILE_DIM (BLOCK_DIM * TILE_FACTOR)
#define BLOCK_DIM 16
#define TILE_FACTOR 2
#define REG_DIM 2

void test_matrix_mul_cpu();
void test_matrix_mul();
