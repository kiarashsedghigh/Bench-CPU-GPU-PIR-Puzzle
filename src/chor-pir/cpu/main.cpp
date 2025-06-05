// This file is part of the CPU implementation of the GF2 matrix-vector multiplication
// for the online server cost of the Chor-PIR.
#include "serverComp.hpp"
#include "misc/helper_timer.h"
#include "misc/helper_string.h"

int main(int argc, char* argv[]) {
    
    size_t r, b, q, iterations;

    r = 1 << (getCmdLineArgumentIntOrDefault(argc, argv, "r", 17));
    b = 1 << (getCmdLineArgumentIntOrDefault(argc, argv, "b", 12));
    q = 1 << (getCmdLineArgumentIntOrDefault(argc, argv, "q", 5));
    iterations = getCmdLineArgumentIntOrDefault(argc, argv, "iterations", 10);
    
    omp_set_num_threads(omp_get_max_threads());

    uint8_t* h_rho      = (uint8_t*)malloc(q * r * sizeof(uint8_t));
    uint8_t* h_DB       = (uint8_t*)malloc(r * b * sizeof(uint8_t));
    uint8_t* h_result   = (uint8_t*)malloc(q * b * sizeof(uint8_t));

    // Load h_DB from file
    FILE* db_file = fopen("../gpu/db.bin", "rb");
    if (db_file) {
        size_t read_bytes = fread(h_DB, sizeof(u8), r * b, db_file);
        if (read_bytes != r * b) {
            fprintf(stderr, "Error: Could not read full DB from file.\n");
            fclose(db_file);
            exit(EXIT_FAILURE);
        }
        fclose(db_file);
    } else {
        fprintf(stderr, "Error: Could not open DB file.\n");
        exit(EXIT_FAILURE);
    }

    // Load h_rho from file
    FILE* rho_file = fopen("../gpu/rho.bin", "rb");
    if (rho_file) {
        size_t read_bytes = fread(h_rho, sizeof(u8), r * q, rho_file);
        if (read_bytes != r * q) {
            fprintf(stderr, "Error: Could not read full DB from file.\n");
            fclose(rho_file);
            exit(EXIT_FAILURE);
        }
        fclose(rho_file);
    } else {
        fprintf(stderr, "Error: Could not open db.bin file.\n");
        exit(EXIT_FAILURE);
    }

    double elapsed_ms = 0;
    for (size_t bench = 0; bench < iterations; bench++) {
        clock_t start = clock();
        gf2_vector_matrix_mult_batch(h_rho, h_DB, h_result, r, b, q);
        clock_t end = clock();

        elapsed_ms += end - start;
    }

    elapsed_ms /= iterations;
    elapsed_ms = (double)elapsed_ms / CLOCKS_PER_SEC * 1000.0;
    printf("\tRUNTIME:: %.3f ms\n", elapsed_ms);

    // Save runtime and algo ID to file
    FILE* fp = fopen("runtime_results.txt", "a");
    if (fp) {
        fprintf(fp, "q::%10d:: b::%10d:: r::%10d ::RUNTIME:: %5.3f ms\n", q, b, r, elapsed_ms);
        fclose(fp);
    } else {
        printf("Error opening file for writing results.\n");
    }

    // printf("\tOUT: ");
    // for (int i = 0; i < 64; i++) {
    //     printf("%u", h_result[i]);
    // }
    // printf("\n");

    // printf("\tOUT: ");
    // for (int i = 64; i < 128; i++) {
    //     printf("%u", h_result[i]);
    // }
    // printf("\n");

    
    free(h_rho);
    free(h_DB);
    free(h_result);

    return 0;
}

//