
#ifndef SERVER_COMP_CUH
#define SERVER_COMP_CUH

#include "util.cuh"
#include "types.h"

/**
 * @brief CUDA Kernel: XOR all rows of DB where rho[i] == 1
 * 
 * @param rho    [r] bytes
 * @param DB     [r][b] bytes
 * @param result [b] bytes
 * @param r      number of rows
 * @param b      number of bytes per record
 * @param Q      number of queries
 */
__global__ void gf2_vector_matrix_mult_basic(
    const u8* __restrict__ rho,    // [r] bytes
    const u8* __restrict__ DB,     // [r][b] bytes
    u8* result,                    // [b] bytes
    u32 r,
    u32 b);

/**
 * @brief CUDA Kernel: Optimized version of gf2_vector_matrix_mult_optimized_1
 * 
 * @param rho    [r] bytes
 * @param DB     [r][b] bytes
 * @param result [b] bytes
 * @param r      number of rows
 * @param b      number of bytes per record
 */
__global__ void gf2_vector_matrix_mult_optimized_1(
    const u8* __restrict__ rho,    // [r] bytes
    const u8* __restrict__ DB,     // [r][b] bytes
    u8* result,                    // [b] bytes
    u32 r,
    u32 b);

/**
 * @brief CUDA Kernel: Optimized version of gf2_vector_matrix_mult_optimized_1 using vectorized operations
 * 
 * @param rho    [r] bytes
 * @param DB     [r][b] bytes
 * @param result [b] bytes
 * @param r      number of rows
 * @param b      number of bytes per record
 */
__global__ void gf2_vector_matrix_mult_optimized_1_finerSync(
    const u8* __restrict__ rho,    // [r] bytes
    const u8* __restrict__ DB,     // [r][b] bytes
    u8* result,                    // [b] bytes
    u32 r,
    u32 b);

/**
 * @brief CUDA Kernel: Optimized version of gf2_vector_matrix_mult_optimized_1 using shfl_xor
 * 
 * @param rho    [r] bytes
 * @param DB     [r][b] bytes
 * @param result [b] bytes
 * @param r      number of rows
 * @param b      number of bytes per record
 */
__global__ void gf2_vector_matrix_mult_shflxor(
    const u8* __restrict__ rho,
    const u32* __restrict__ DB,
    u32* result,
    u32 r,
    u32 b);

/**
 * @brief CUDA Kernel: Optimized version of gf2_vector_matrix_mult_optimized_1 using vectorized uint4 operations
 * 
 * @param rho    [r] bytes
 * @param DB     [r][b] bytes
 * @param result [b] bytes
 * @param r      number of rows
 * @param b      number of bytes per record
 */
__global__ void gf2_vector_matrix_mult_optimized_1_uint4(
    const u8* __restrict__ rho,    // [r] bytes
    const uint4* __restrict__ DB,     // [r][b] bytes
    uint4* result,                    // [b] bytes
    u32 r,
    u32 b);

/**
 * @brief CUDA Kernel: Optimized version of gf2_vector_matrix_mult_optimized_1 using vectorized ulong4 operations
 * 
 * @param rho    [r] bytes
 * @param DB     [r][b] bytes
 * @param result [b] bytes
 * @param r      number of rows
 * @param b      number of bytes per record
 */
__global__ void gf2_vector_matrix_mult_optimized_1_ulong4(
    const u8* __restrict__ rho,    // [r] bytes
    const ulong4* __restrict__ DB,     // [r][b] bytes
    ulong4* result,                    // [b] bytes
    u32 r,
    u32 b);

/**
 * @brief CUDA Kernel: Optimized version of gf2_vector_matrix_mult_optimized_1 using shared mnemory to load DB chunks
 * 
 * @param rho    [r] bytes
 * @param DB     [r][b] bytes
 * @param result [b] bytes
 * @param r      number of rows
 * @param b      number of bytes per record
 */
__global__ void gf2_vector_matrix_mult_optimized_2(
    const u8* __restrict__ rho,
    const u8* __restrict__ DB,
    u8* result,
    u32 r,
    u32 b);

/**
 * @brief CUDA Kernel: Optimized version of gf2_vector_matrix_mult_optimized_3 for batch processing
 * 
 * @param rho    [Q][r] bytes
 * @param DB     [r][b] bytes
 * @param result [Q][b] bytes
 * @param r      number of rows
 * @param b      number of bytes per record
 * @param Q      number of queries
 */
__global__ void gf2_vector_matrix_mult_batch(
    const u8* __restrict__ rho,     // [q][r]
    const u8* __restrict__ DB,      // [r][b]
    u8* result,                     // [q][b]
    u32 r, 
    u32 b,
    u32 q);

__global__ void gf2_vector_matrix_mult_batch_u32(
    const u8* __restrict__ rho,      // [q][r]
    const u32* __restrict__ DB,      // [r][b]
    u32* result,                     // [q][b]
    u32 r,
    u32 b,
    u32 q);

__global__ void gf2_vector_matrix_mult_batch_uint4(
    const u8* __restrict__ rho,      // [q][r]
    const uint4* __restrict__ DB,      // [r][b]
    uint4* result,                     // [q][b]
    u32 r,
    u32 b,
    u32 q);

__global__ void gf2_vector_matrix_mult_batch_uint4_shflxor(
    const u8* __restrict__ rho,        // [q][r]
    const uint4* __restrict__ DB,      // [r][b]
    uint4* result,                     // [q][b]
    u32 r,
    u32 b,
    u32 q
);

__global__ void gf2_vector_matrix_mult_batch_ulong4(
    const u8* __restrict__ rho,      // [q][r]
    const ulong4* __restrict__ DB,      // [r][b]
    ulong4* result,                     // [q][b]
    u32 r,
    u32 b,
    u32 q);
#endif // SERVER_COMP_CUH