#include <sstream>
#include <iostream>
#include <NTL/LLL.h>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>

NTL_CLIENT;

#include "tools.h"

void thread_worker(int thread_id, int num_iters, long n, long bit, ZZ seed)
{
    vec_ZZ v;
    ZZ thread_seed = seed + thread_id; // ensure different seed per thread if needed
    for (int i = 0; i < num_iters; ++i) {
        generate_random_HNF(v, n, bit, thread_seed);
    }
}

int main(int argc, char** argv)
{
    long n = 80;
    long bit = 20;
    ZZ seed; seed = 0;

    PARSE_MAIN_ARGS {
        MATCH_MAIN_ARGID("--dim", n);
        MATCH_MAIN_ARGID("--seed", seed);
        // MATCH_MAIN_ARGID("--bit", bit);
        SYNTAX();
    }

    int count = 1 << 12;
    int num_threads = std::thread::hardware_concurrency(); // or set manually

    int per_thread = count / num_threads;
    int leftover = count % num_threads;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        int iters = per_thread + (i < leftover ? 1 : 0); // distribute leftover
        threads.emplace_back(thread_worker, i, iters, n, bit, seed);
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << duration.count() << "ms" << std::endl;

    return 0;
}
