#ifndef UTIL_CUH
#define UTIL_CUH

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "types.h"
#include "misc/helper_string.h"
#include "misc/helper_timer.h"

#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

/**
 * @brief Structure to hold matrix information
 * 
 * info: 
 * argc: Number of command line arguments
 * argv: Command line arguments
 */
u32 matrix_info_init(matrix_info_t* info, int argc, char** argv);

/**
 * @brief Structure to hold matrix data
 * 
 * data: 
 * info: Matrix information (rows, columns, queries)
 */
u32 matrix_data_init(matrix_data_t* data, matrix_info_t* info);

/**
 * @brief Free the memory allocated for matrix data
 * 
 * data: Pointer to the matrix data structure
 * 
 * Returns: Number of bytes freed
 */
u32 matrix_data_free(matrix_data_t* data);

#endif // UTIL_CUH