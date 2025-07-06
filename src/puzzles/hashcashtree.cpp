#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cstdlib>
#include <openssl/sha.h>

#define MAX_INPUT_SIZE 128

using namespace std;

// === Fake SHA-256 (Randomized but deterministic for testing) ===
void sha256(const char* input, int len, unsigned char* output) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, input, len);
    SHA256_Final(output, &ctx);
}

// === Check k leading zero bits ===
bool has_k_leading_zeros(const unsigned char* hash, int k) {
    int bits = 0;
    for (int i = 0; i < 32 && bits < k; ++i) {
        unsigned char byte = hash[i];
        for (int j = 7; j >= 0 && bits < k; --j) {
            if ((byte >> j) & 1)
                return false;
            bits++;
        }
    }
    return true;
}

// === Multithreaded CPU Hashcash ===
void hashcash_cpu_mt(const string& input_str, int k, int& nonce_out, unsigned char* hash_out, int thread_count = 8) {
    atomic<bool> found(false);
    atomic<int> nonce(-1);
    vector<thread> threads;
    mutex hash_mutex;

    auto worker = [&](int tid) {
        int x = tid;
        unsigned char local_hash[32];
        char input[MAX_INPUT_SIZE];

        while (!found.load()) {
            int len = snprintf(input, MAX_INPUT_SIZE, "%s%d", input_str.c_str(), x);
            sha256(input, len, local_hash);

            if (has_k_leading_zeros(local_hash, k)) {
                if (!found.exchange(true)) {
                    nonce = x;
                    lock_guard<mutex> lock(hash_mutex);
                    memcpy(hash_out, local_hash, 32);
                }
                return;
            }
            x += thread_count;
        }
    };

    for (int t = 0; t < thread_count; ++t)
        threads.emplace_back(worker, t);

    for (auto& t : threads)
        t.join();

    nonce_out = nonce.load();
}

// === Build Hashcash Tree ===
void hashcash_tree(const string& s, int n, int k, int thread_count = 8) {
    int tree_size = 2 * n;

    // Allocate hash tree and nonce array on heap
    unsigned char (*tree_hash)[32] = (unsigned char (*)[32])malloc(sizeof(unsigned char[32]) * tree_size);
    int* tree_nonce = (int*)malloc(sizeof(int) * tree_size);

    if (!tree_hash || !tree_nonce) {
        cerr << "Memory allocation failed. Try reducing tree size.\n";
        exit(EXIT_FAILURE);
    }

    // Step 1: Initialize leaves using H("e" || i)
    for (int i = n; i < tree_size; ++i) {
        char leaf_input[MAX_INPUT_SIZE];
        int len = snprintf(leaf_input, sizeof(leaf_input), "e%d", i);
        sha256(leaf_input, len, tree_hash[i]);
        tree_nonce[i] = 0;
    }

    // Step 2: Build internal nodes bottom-up
    for (int i = n - 1; i >= 1; --i) {
        string input = s + to_string(i);
        input.append((char*)tree_hash[2 * i], 32);
        input.append((char*)tree_hash[2 * i + 1], 32);

        int nonce;
        unsigned char hash[32];
        hashcash_cpu_mt(input, k, nonce, hash, thread_count);

        memcpy(tree_hash[i], hash, 32);
        tree_nonce[i] = nonce;

        cout << "Node " << i << " done (nonce=" << nonce << ")\n";
    }

    // Step 3: Print root hash
    cout << "Root hash: ";
    for (int i = 0; i < 32; ++i)
        printf("%02x", tree_hash[1][i]);
    cout << endl;

    // Free memory
    free(tree_hash);
    free(tree_nonce);
}

// === Main ===
int main() {
    string s = "OpenAI";
    int n = 1 << 12;     // 16384 leaves
    int k = 20;          // Difficulty: leading zero bits
    int threads = 16;

    cout << "Building HASHCASH TREE for " << n << " leaves using " << threads << " threads...\n";

    auto start = chrono::high_resolution_clock::now();
    hashcash_tree(s, n, k, threads);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "Execution time: " << duration.count() << " microseconds\n";

    return 0;
}
