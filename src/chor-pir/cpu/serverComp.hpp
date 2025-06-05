
#ifndef CPU_SERVERCOMP_HPP
#define CPU_SERVERCOMP_HPP

#include "util.h"

/**
 * @brief Multiplies a binary vector with a binary matrix.
 * 
 * rho: The binary vector of size `r`.
 * DB: The binary matrix of size `r x b`.
 * result: The output binary vector of size `b`.
 * r: The number of rows in the matrix.
 * b: The number of columns in the matrix.
 */
void gf2_vector_matrix_mult_basic(uint8_t* rho, 
                                   uint8_t* DB,
                                   uint8_t* result,
                                   int r,
                                   int b);

/**
 * @brief Multiplies a binary vector with a binary matrix.
 * 
 * rho: The binary vector of size `r`.
 * DB: The binary matrix of size `r x b`.
 * result: The output binary vector of size `b`.
 * r: The number of rows in the matrix.
 * b: The number of columns in the matrix.
 */
void gf2_vector_matrix_mult_optimized(uint8_t* rho,
                                   uint8_t* DB,
                                   uint8_t* result,
                                   int r,
                                   int b);

/**
 * @brief Multiplies a binary vector with a binary matrix in parallel.
 * 
 * rho: The binary vector of size `r`.
 * DB: The binary matrix of size `r x b`.
 * result: The output binary vector of size `b`.
 * r: The number of rows in the matrix.
 * b: The number of columns in the matrix.
 */
void gf2_vector_matrix_mult_batch(uint8_t* rho,
                                  uint8_t* DB,
                                  uint8_t* result,
                                  int r,
                                  int b,
                                  size_t Q);

#endif // CPU_SERVERCOMP_HPP