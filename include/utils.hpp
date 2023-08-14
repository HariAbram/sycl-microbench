

#ifndef UTILS_H
#define UTILS_H

#define BLOCK_SIZE 16
#define SEED 18945678

typedef enum _FUNC_RETURN_CODE {
    RET_SUCCESS,
    RET_FAILURE
}func_ret_t;

func_ret_t create_matrix_from_file(float **mp, const char* filename, int *size_p);
func_ret_t create_matrix(float *__restrict__ *mp, int size);
func_ret_t create_vector(float **vp, int size);
func_ret_t create_sparse_matrix(float **mp, int size);
func_ret_t create_sparse_matrix_from_file(int **rp, int **cp, float **vp, const char* filename, int *nnz, int *size_p);




#endif