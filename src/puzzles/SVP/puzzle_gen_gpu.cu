#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Simulated "prime" function (same as before)
__device__ unsigned int fake_prime(int bit_n) {
    // Simulate 50 ms delay
    unsigned long long delay_cycles = 175000000ULL;
    unsigned long long start = clock64();
    while (clock64() - start < delay_cycles) {
        // busy wait
    }

    return (1U << (bit_n - 1)) + 1U;
}




// Kernel using cuRAND to generate random numbers
__global__ void simulate_generate_random_HNF(int* output, int n, int bit, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Set up cuRAND RNG state for this thread
    curandState state;
    curand_init(1234ULL, tid, 0, &state); // seed, sequence id, offset

    for (int idx = tid; idx < count; idx += total_threads) {
        int prime = fake_prime(bit * n);
        output[idx * n + 0] = prime;

        for (int i = 1; i < n; ++i) {
            float r = curand_uniform(&state);       // float ∈ (0,1]
            for(int i=1; i < n; ++i) // simulating random n * b-bit number
                r += curand_uniform(&state);
            int rand_val = (int)(r * prime);        // scale to [0, prime)
            output[idx * n + i] = rand_val;
        }
    }
}


int main() {
    int n = 79;
    int bit = 10;
    int count = 1 << 10;

    int* d_output;
    cudaMalloc((void**)&d_output, sizeof(int) * n * count);

    int threads_per_block = 1024;
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
