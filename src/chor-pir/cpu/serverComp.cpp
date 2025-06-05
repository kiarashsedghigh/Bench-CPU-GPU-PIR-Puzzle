
#include "serverComp.hpp"

void gf2_vector_matrix_mult_basic(uint8_t* rho, 
                                   uint8_t* DB,
                                   uint8_t* result,
                                   int r,
                                   int b) {
    memset(result, 0, b);

    for (int i = 0; i < r; i++) {
            if (rho[i] == 1) {
                for (int j = 0; j < b; j++)
                    result[j] ^= DB[i * b + j];
            }
    }
}

void gf2_vector_matrix_mult_optimized(uint8_t* rho, 
                                   uint8_t* DB,
                                   uint8_t* result,
                                   int r,
                                   int b) {
    memset(result, 0, b);

    for (int i = 0; i < r; i++) {
        if (rho[i]) {
            for (int j = 0; j < b; j += 32) {
                __m256i rvec = _mm256_loadu_si256((__m256i*)&result[j]);
                __m256i dvec = _mm256_loadu_si256((__m256i*)&DB[i * b + j]);
                rvec = _mm256_xor_si256(rvec, dvec);
                _mm256_storeu_si256((__m256i*)&result[j], rvec);
            }
        }
    }
}

void gf2_vector_matrix_mult_batch(uint8_t* rho,
                                  uint8_t* DB,
                                  uint8_t* result,
                                  int r,
                                  int b,
                                  size_t q) {
    memset(result, 0, q * b);  // Ensure output is zero-initialized

    // Parallelize over the outer query dimension
    #pragma omp parallel for schedule(static)
    for (size_t query = 0; query < q; query++) {
        gf2_vector_matrix_mult_optimized(rho + query * r, 
                                         DB, 
                                         result + query * b, 
                                         r, 
                                         b);
    }
}
