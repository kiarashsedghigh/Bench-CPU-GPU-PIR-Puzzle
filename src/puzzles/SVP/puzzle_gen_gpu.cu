#include <stdio.h>
#include <cuda_runtime.h>

__device__ unsigned long long fake_seed(int thread_id, int iteration) {
    return 123456789ULL + thread_id * 65537ULL + iteration * 997ULL;
}

__device__ unsigned int fake_prime(int bit_n) {
    return (1U << (bit_n - 1)) + 1U;  // fake "prime"
}

__device__ unsigned int fake_rand(unsigned int mod, unsigned long long seed) {
    return (unsigned int)((seed ^ 0x5DEECE66DULL) % mod);
}

__global__ void simulate_generate_random_HNF(int* output, int n, int bit, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    unsigned long long seed = fake_seed(idx, 0);
    int prime = fake_prime(bit * n);

    output[idx * n + 0] = prime;

    for (int i = 1; i < n; ++i) {
        output[idx * n + i] = fake_rand(prime, seed + i);
    }
}

int main() {
    int n = 80;
    int bit = 20;
    int count = 1 << 11;

    int* d_output;
    cudaMalloc((void**)&d_output, sizeof(int) * n * count);

    int threads_per_block = 256;
    int blocks = (count + threads_per_block - 1) / threads_per_block;

    // Timing using CUDA events (C-style)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    simulate_generate_random_HNF<<<blocks, threads_per_block>>>(d_output, n, bit, count);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU simulation duration: %.2f ms\n", milliseconds);

    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
