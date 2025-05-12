#include "mm_mult.cuh"

#include <iostream>
#include <vector>


int main() {

    const int M = 1, N = 512, K = 1<<20;
    const int size_A = M * K;
    const int size_B = K * N;
    const int size_C = M * N;

    // Host allocations
    std::vector<float> h_A(size_A), h_B(size_B), h_C_cpu(size_C), h_C_gpu(size_C);

    // Initialize A and B
    for (int i = 0; i < size_A; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < size_B; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // // CPU timing
    // auto start_cpu = std::chrono::high_resolution_clock::now();
    // cpu_matmul(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
    // auto end_cpu = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    // std::cout << "CPU Time: " << cpu_duration.count() << " seconds\n";

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C * sizeof(float));

    // Define grid and block dimensions
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));

    // GPU timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    cuda_mm_mult<float><<<gridDim, blockDim>>>(M, K, N, d_A, d_B, d_C);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    std::cout << "GPU Time: " << gpu_time / 1000.0f << " seconds\n";

    // Copy back and compare
    cudaMemcpy(h_C_gpu.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);

    // if (compare_matrices(h_C_cpu.data(), h_C_gpu.data(), size_C)) {
    //     std::cout << "Results match!\n";
    // } else {
    //     std::cerr << "Mismatch detected in results.\n";
    // }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

