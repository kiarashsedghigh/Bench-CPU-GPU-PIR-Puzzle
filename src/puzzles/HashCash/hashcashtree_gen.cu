#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <vector>
#include <iostream>

#include "sha2.cuh"  // Assumes you have a SHA256_CTX_GPU-compatible implementation

#define INPUT_SIZE 32
#define HASH_SIZE 32

__device__ void sha256_gpu(const char* input, int len, unsigned char* output) {
    SHA256_CTX_GPU ctx;
    sha256_init_gpu(&ctx);
    sha256_update_gpu(&ctx, (const BYTE*)input, len);
    sha256_final_gpu(&ctx, output);
}

__global__ void hashcashtree_gen_puzzle(char* d_inputs, unsigned char* d_hashes, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    char input[INPUT_SIZE];
    unsigned char hash[HASH_SIZE];

    // Load input
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = d_inputs[tid * INPUT_SIZE + i];
    }

    // Compute SHA-256
    sha256_gpu(input, INPUT_SIZE, hash);

    // Store result
//    for (int i = 0; i < HASH_SIZE; i++) {
//        d_hashes[tid * HASH_SIZE + i] = hash[i];
//   }
}

int main() {
    const int count = 1 << 12;
    const int total_input_bytes = count * INPUT_SIZE;
    const int total_hash_bytes = count * HASH_SIZE;

    // Allocate host memory
    std::vector<char> h_inputs(total_input_bytes);
    std::vector<unsigned char> h_hashes(total_hash_bytes);

    // Fill input with deterministic data
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            h_inputs[i * INPUT_SIZE + j] = (char)(i + j);
        }
    }

    // Allocate GPU memory
    char* d_inputs;
    unsigned char* d_hashes;
    cudaMalloc(&d_inputs, total_input_bytes);
    cudaMalloc(&d_hashes, total_hash_bytes);

    cudaMemcpy(d_inputs, h_inputs.data(), total_input_bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (count + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event just before kernel launch
    cudaEventRecord(start);

    // Launch the kernel
    hashcashtree_gen_puzzle<<<blocks, threads_per_block>>>(d_inputs, d_hashes, count);

    // Record stop event just after kernel execution
    cudaEventRecord(stop);

    // Wait for kernel to finish and synchronize
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %.2f ms\n", ms);

    // Copy hashes back to host (optional)
    cudaMemcpy(h_hashes.data(), d_hashes, total_hash_bytes, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_inputs);
    cudaFree(d_hashes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
