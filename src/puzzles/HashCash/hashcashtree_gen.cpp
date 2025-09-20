#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <openssl/sha.h>
#include <cstring>   // for memset

void sha256(const char* input, int len, unsigned char* output) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, input, len);
    SHA256_Final(output, &ctx);
}

int x {};

void thread_worker(int iters) {
    char in[32];
    unsigned char out[32];

    for (int i = 0; i < iters; i++) {
        sha256(in, 32, out);
        x += out[0];
        // Optionally: use `out` for something to prevent compiler optimization
        // volatile to prevent optimization
        volatile unsigned char sink = out[0];
        (void)sink;
    }
}

int main(int argc, char** argv) {
    const int total_hashes = 1 << 16;
    const int num_threads = 1;// std::thread::hardware_concurrency(); // auto-detect cores


    int per_thread = total_hashes / num_threads;
    int leftover = total_hashes % num_threads;

    std::vector<std::thread> threads;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_threads; ++i) {
        int iters = per_thread + (i < leftover ? 1 : 0);
        threads.emplace_back(thread_worker, iters);
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Hashed " << total_hashes << " inputs using " << num_threads
              << " threads in " << duration_ms << "micro s" << std::endl;
    return 0;
}
