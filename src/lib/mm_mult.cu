#include <iostream>
#include <chrono>
#include "mm_mult.cuh"

// // CPU version of matrix multiplication
// void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
//     for (int row = 0; row < M; ++row) {
//         for (int col = 0; col < N; ++col) {
//             float sum = 0.0f;
//             for (int k = 0; k < K; ++k) {
//                 sum = (int)((sum + A[row * K + k] * B[k * N + col])) % Q;
//             }
//             C[row * N + col] = sum;
//         }
//     }
// }
//
// // Compare CPU and GPU results
// bool compare_matrices(const float* A, const float* B, int size, float eps = 1e-2f) {
//     for (int i = 0; i < size; ++i) {
//         if (std::fabs(A[i] - B[i]) > eps) {
//             std::cerr << "Mismatch at index " << i << ": " << A[i] << " vs. " << B[i] << '\n';
//             return false;
//         }
//     }
//     return true;
// }


