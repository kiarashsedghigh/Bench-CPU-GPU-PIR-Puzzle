#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <openssl/sha.h>
#include <sys/time.h>
#include "sha2.cuh"

#define THREADS_PER_BLOCK 256
#define MAX_INPUT_SIZE 256
#define HASH_SIZE SHA256_DIGEST_LENGTH

// === CPU SHA256 ===
void sha256(const char* input, int len, unsigned char* output) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, input, len);
    SHA256_Final(output, &ctx);
}

// === GPU SHA256 wrappers ===
__device__ void sha256_gpu(const char* input, int len, unsigned char* output) {
    SHA256_CTX_GPU ctx;
    sha256_init_gpu(&ctx);
    sha256_update_gpu(&ctx, (const BYTE*)input, len);
    sha256_final_gpu(&ctx, output);
}

// === GPU: Check k leading zero bits ===
__device__ bool has_k_leading_zeros(const unsigned char* hash, int k) {
    int bits = 0;
    for (int i = 0; i < HASH_SIZE && bits < k; ++i) {
        unsigned char byte = hash[i];
        for (int j = 7; j >= 0 && bits < k; --j) {
            if ((byte >> j) & 1)
                return false;
            bits++;
        }
    }
    return true;
}

__global__ void hashcash_kernel(const char* base, int base_len, int k, int* result_nonce, unsigned char* result_hash, int* found) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    char input[MAX_INPUT_SIZE];
    unsigned char hash[HASH_SIZE];

    for (int nonce = idx; ; nonce += total_threads) {
        if (atomicAdd(found, 0)) return;

        int pos = 0;

        // Copy base input
        for (int i = 0; i < base_len && pos < MAX_INPUT_SIZE; ++i)
            input[pos++] = base[i];

        // Append nonce digits
        int val = nonce;
        char digits[20];  // large enough for 64-bit ints
        int d = 0;
        do {
            digits[d++] = '0' + (val % 10);
            val /= 10;
        } while (val > 0);

        for (int i = d - 1; i >= 0 && pos < MAX_INPUT_SIZE; --i)
            input[pos++] = digits[i];

        // Hash it
        sha256_gpu(input, pos, hash);

        // Check leading zeros
        if (has_k_leading_zeros(hash, k)) {
            if (atomicCAS(found, 0, 1) == 0) {
                *result_nonce = nonce;
                for (int i = 0; i < HASH_SIZE; ++i)
                    result_hash[i] = hash[i];
            }
            return;
        }
    }
}

// === Host GPU Hashcash ===
void hashcash_gpu(const char* input_str, int k, int* out_nonce, unsigned char* out_hash) {
    char* d_input;
    int* d_nonce;
    unsigned char* d_hash;
    int* d_found;

    int input_len = strlen(input_str);

    cudaMalloc(&d_input, MAX_INPUT_SIZE);
    cudaMemcpy(d_input, input_str, input_len, cudaMemcpyHostToDevice);

    cudaMalloc(&d_nonce, sizeof(int));
    cudaMalloc(&d_hash, HASH_SIZE);
    cudaMalloc(&d_found, sizeof(int));

    cudaMemset(d_nonce, 0, sizeof(int));
    cudaMemset(d_hash, 0, HASH_SIZE);
    cudaMemset(d_found, 0, sizeof(int));

    int blocks = 4096;
    hashcash_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, input_len, k, d_nonce, d_hash, d_found);
    cudaDeviceSynchronize();

    cudaMemcpy(out_nonce, d_nonce, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_hash, d_hash, HASH_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_nonce);
    cudaFree(d_hash);
    cudaFree(d_found);
}

// === Build Hashcash Tree using GPU ===
void hashcash_tree(const char* s, int n, int k) {
    int tree_size = 2 * n;
    unsigned char (*tree_hash)[HASH_SIZE];
    int* tree_nonce;

    cudaMallocManaged(&tree_hash, sizeof(unsigned char[HASH_SIZE]) * (tree_size + 1));
    cudaMallocManaged(&tree_nonce, sizeof(int) * (tree_size + 1));

    if (!tree_hash || !tree_nonce) {
        fprintf(stderr, "CUDA malloc failed\n");
        exit(1);
    }

    // Step 1: Initialize leaf nodes
    for (int i = n; i <= 2 * n - 1; ++i) {
        char leaf_input[MAX_INPUT_SIZE];
        snprintf(leaf_input, MAX_INPUT_SIZE, "e%d", i);
        sha256(leaf_input, strlen(leaf_input), tree_hash[i]);
        tree_nonce[i] = 0;
    }

    // Step 2: Compute internal nodes
    for (int i = n - 1; i >= 1; --i) {
        char input[MAX_INPUT_SIZE];
        int pos = snprintf(input, MAX_INPUT_SIZE, "%s%d", s, i);

        if (pos + 2 * HASH_SIZE > MAX_INPUT_SIZE) {
            fprintf(stderr, "Buffer overflow at node %d\n", i);
            exit(1);
        }

        memcpy(input + pos, tree_hash[2 * i], HASH_SIZE);
        memcpy(input + pos + HASH_SIZE, tree_hash[2 * i + 1], HASH_SIZE);
        int input_len = pos + 2 * HASH_SIZE;

        int nonce;
        unsigned char hash[HASH_SIZE];
        hashcash_gpu(input, k, &nonce, hash);

        memcpy(tree_hash[i], hash, HASH_SIZE);
        tree_nonce[i] = nonce;

        printf("Node %d done (nonce=%d)\n", i, nonce);
    }

    // Step 3: Print root hash
//    printf("Root hash: ");
    for (int i = 0; i < HASH_SIZE; ++i)
        printf("%02x", tree_hash[1][i]);
    printf("\n");

    cudaFree(tree_hash);
    cudaFree(tree_nonce);
}

// === Main ===
int main() {
    const char* s = "OpenAI";
    int n = 1 << 12;       // Number of leaves (must be power of 2)
    int k = 20;      // Difficulty level

    printf("Building HASHCASH TREE for %d leaves with k=%d...\n", n, k);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    hashcash_tree(s, n, k);

    gettimeofday(&end, NULL);
    long us = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
    printf("Execution time: %ld microseconds\n", us);

    return 0;
}
