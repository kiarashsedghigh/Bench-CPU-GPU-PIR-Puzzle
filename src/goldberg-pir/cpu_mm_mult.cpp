#include <stdio.h>
#include <sys/time.h>
#include <immintrin.h>
#include <ctime>
#include <cmath>
#include <cstring>
#include <iostream>

#include <chrono>

#define CEIL(M, N) (((M) + (N)-1) / (N))
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/* Macros for changing the datatype from float to double easily in our code */
#define DATA_TYPE int // float for more efficiency > double

/* DEBUG_MODE macro allows for managing the details that we want to print in the console */
// #define DEBUG_MODE

/* Matrix is 2D matrix of type T stored in the row-major fashion */
template<typename T>
struct Matrix {
    T *data;
};


/**
 * \brief Creates a MxN matrix stored in row-major fashion
 * \param matrix: Pointer to the matrix
 * \param M: Row dimension
 * \param N: Column dimension
 * \param Q: Modulus (optional)
 */
template<typename T>
void create_matrix(Matrix<T> *matrix, int M, int N, int Q = 0) {
    matrix->data = (T *) malloc(sizeof(T) * M * N);

    // Random seed set to the time of the system
    srand(time(NULL));

    for (int i = 0; i < M * N; i++) {
        auto temp_value = rand();
        if (Q)
            temp_value = temp_value % Q;
        matrix->data[i] = (T) temp_value;
    }
}

/**
 * \brief Creates a zeroed MxN matrix stored in row-major fashion
 * \param matrix: Pointer to the matrix
 * \param M: Row dimension
 * \param N: Column dimension
 */
template<typename T>
void create_zero_matrix(Matrix<T> *matrix, int M, int N) {
    matrix->data = (T *) malloc(sizeof(T) * M * N);
    for (int i = 0; i < M * N; i++)
        matrix->data[i] = 0;
}


/**
 * \brief Deletes the matrix by freeing its dynamic memory allocated
 * \param matrix: Pointer to the matrix
 */
template<typename T>
void delete_matrix(Matrix<T> *matrix) {
    free(matrix->data);
}

/**
 * \brief This template function prints a matrix of size MxN.
 * \param data: Pointer to the actual matrix data of type T
 * \param M: Row dimension
 * \param N: Column dimension
 * \param name: Matrix's name
 */
template<typename T>
void print_matrix(Matrix<T> *matrix, const int M, const int N, char *name) {
    if (name)
        printf("%s = [ ", name);
    else
        printf("[ ");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%f ", matrix->data[i * N + j]);
        if (i == M - 1)
            printf("]\n");
        else
            printf("\n");
    }
}


/**
 * \brief CPU-based implementation of Matrix-Matrix multiplication (C_mn = A_mk * B_kn)
 * \param M: Row dimension of A
 * \param K: Column and Row dimension of A and B, respectively
 * \param N: Column dimension of B
 * \param A: Input matrix A
 * \param B: Input matrix B
 * \param C: Output matrix C
 * \param Q: Modulus
 */
// Transpose B[K x N] into B_T[N x K]
void transpose_B(const int32_t* __restrict B, int32_t* __restrict B_T, int K, int N) {
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            B_T[n * K + k] = B[k * N + n];
}

template<typename T>
void cpu_matrix_matrix_multiplier(int M, int K, int N,
                                  const T* __restrict A,
                                  const T* __restrict B,
                                  T* __restrict C,
                                  int Q = 0) {
    static_assert(std::is_same<T, int32_t>::value, "Only int32_t is supported.");

    // Transpose B
    int32_t* B_T = new int32_t[N * K];
    transpose_B(reinterpret_cast<const int32_t*>(B), B_T, K, N);

    alignas(32) int32_t temp_acc[8];

    for (int i = 0; i < M; ++i) {
        const int32_t* a_ptr = reinterpret_cast<const int32_t*>(A + i * K);
        int32_t* c_ptr = reinterpret_cast<int32_t*>(C + i * N);

        for (int j = 0; j <= N - 8; j += 8) {
            int64_t acc[8] = {};

            for (int k = 0; k < K; ++k) {
                int32_t a_val = a_ptr[k];
                const int32_t* b_row = &B_T[j * K + k];

                for (int t = 0; t < 8; ++t) {
                    acc[t] += static_cast<int64_t>(a_val) * static_cast<int64_t>(b_row[t * K]);
                }
            }

            for (int t = 0; t < 8; ++t) {
                int64_t val = acc[t];
                if (Q != 0)
                    val = ((val % Q) + Q) % Q;
                c_ptr[j + t] = static_cast<int32_t>(val);
            }
        }

        // Tail: N not divisible by 8
        for (int j = (N / 8) * 8; j < N; ++j) {
            int64_t acc = 0;
            for (int k = 0; k < K; ++k) {
                int32_t a_val = a_ptr[k];
                int32_t b_val = B_T[j * K + k];
                acc += static_cast<int64_t>(a_val) * b_val;
            }
            if (Q != 0)
                acc = ((acc % Q) + Q) % Q;
            c_ptr[j] = static_cast<int32_t>(acc);
        }
    }

    delete[] B_T;
}

/**
 * \brief Computes the timing of an operation based on the passed time structs
 * \param startTime: Pointer to the start time struct
 * \param endTime: Pointer to the end time struct
 * \param timeZone: Pointer to the time zone struct
 */
void report_running_time(struct timeval *startTime, struct timeval *endTime, struct timezone *timeZone) {
    long sec_diff, usec_diff;
    gettimeofday(endTime, timeZone);
    sec_diff = endTime->tv_sec - startTime->tv_sec;
    usec_diff = endTime->tv_usec - startTime->tv_usec;
    if (usec_diff < 0) {
        sec_diff--;
        usec_diff += 1000000;
    }
    printf("Running time for %s: %ld.%06ld (s)\n", "CPU", sec_diff, usec_diff);
}


/**
 * \brief Runs the Goldberg's PIR server in CPU mode
 * \param M: Row dimension of A
 * \param K: Column and Row dimension of A and B, respectively
 * \param N: Column dimension of B
 * \param A: Pointer to input Matrix A
 * \param B: Pointer to input Matrix B
 * \param C: Pointer to output Matrix C
 * \param Q: Modulus
 */
template<typename T>
void server_cpu(int M, int K, int N, Matrix<T> *A, Matrix<T> *B, Matrix<T> *C, int Q = 0) {
    printf("Sever is running on CPU ...\n");
    struct timezone tz;
    struct timeval startTime, endTime;
    gettimeofday(&startTime, &tz);

    /* Run the matrix-matrix multiplication iAn CPU mode. Precisely, in PIR,
    this is vector-matrix multiplication as M=1 */
    cpu_matrix_matrix_multiplier<T>(M, K, N, A->data, B->data, C->data, Q);

    gettimeofday(&endTime, &tz);
    report_running_time(&startTime, &endTime, &tz);
}

/**
 * \brief Returns multiplicative inverse of a number in the field
 * \param a: Input number
 * \param q: Modulus q of the field Z_q
 */
long modinv(long a, long q) {
    long t = 0, new_t = 1;
    long r = q, new_r = a;

    while (new_r != 0) {
        long quotient = r / new_r;
        long temp_t = t;
        t = new_t;
        new_t = temp_t - quotient * new_t;
        long temp_r = r;
        r = new_r;
        new_r = temp_r - quotient * new_r;
    }

    if (r > 1) {
        printf("No modular inverse exists for %ld mod %ld\n", a, q);
        exit(-1);
    }
    if (t < 0)
        t += q;

    return t;
}


/**
 * \brief Performs lagranage interpolation at 0. I.e., it computes f(0)
 * \param x: List of X coordinates
 * \param y: List of Y coordinates
 * \param t: Number of (X,Y) coordinates
 * \param q: Modulus q of the field Z_q
 */
long lagrange_interpolate_at_zero(const long *x, const long *y, int t, long q) {
    long result = 0;

    for (int i = 0; i <= t; i++) {
        long num = 1;
        long denom = 1;

        for (int j = 0; j <= t; j++) {
            if (j == i) continue;
            num = (num * (-x[j] + q)) % q;
            denom = (denom * (x[i] - x[j] + q)) % q;
        }

        long li = (num * modinv(denom, q)) % q;
        result = (result + y[i] * li) % q;
    }

    return result;
}


/**
 * \brief EasyRocover algorithm of the Goldberg's PIR
 *   See: Ian Goldberg. Improving the robusTHREAD_LOCAL_Ness of private information retrieval. In IEEE Symposium on
        Security and Privacy (SP’07). IEEE, 2007.
 * \param alphas: Evaluation points
 * \param responses: Array of responses from the server
 * \param recovered_block: Pointer to the recovered block that we will write into it after recovering
 * \param N: Column dimension of B
 * \param L: Parameter L of the PIR (#Servers)
 * \param T: Parameter T of the PIR
 * \param Q: Modulus
 */
template<typename U>
void goldberg_pir_easy_recover(long *alphas, Matrix<U> responses[], Matrix<long> *recovered_block, int N, int L, int T,
                               int Q) {
    /* From now on, we can rely on integer type variables like long. We follow EasyReecover Algorithm from Goldberg's paper. */

    // Allocating a vector of size 1xL to record L responses for each column of the desired block
    Matrix<long> easy_recover_vector;
    create_zero_matrix<long>(&easy_recover_vector, 1, L);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < L; j++)
            easy_recover_vector.data[j] = (long) responses[j].data[i];
        recovered_block->data[i] = lagrange_interpolate_at_zero(alphas, easy_recover_vector.data, T, Q);
    }

#ifdef DEBUG_MODE
    printf("Block: [ ");
    for (int i = 0; i < N; i++)
        printf("%ld ", recovered_block->data[i]);
    printf("]\n");
#endif
}


void run(const int M, const int K, const int N, const long INDEX, const int Q, const int L, const int T) {
    printf("----------- Input INFO -----------\n");
    printf("Matrix Dimensions: %dx%d\n", K, N);
    printf("Query Dimensions: %dx%d\n", M, K);
    printf("Tolerance (t, l) = (%d , %d)\n", T, L);
    printf("Chosen Modulus: %d\n", Q);


    printf("\n----------- Creating Database -----------\n");
    Matrix<DATA_TYPE> db;
    create_matrix<DATA_TYPE>(&db, K, N, Q);

#ifdef DEBUG_MODE
    print_matrix<DATA_TYPE>(&db, K, N, (char *) "DB");
#endif


    struct timezone tz;
    struct timeval startTime, endTime;
    gettimeofday(&startTime, &tz);


    printf("\n----------- Creating Query for Index %ld -----------\n", INDEX);
    Matrix<DATA_TYPE> query[L]; // L queries for L servers
    long alphas[L]; // Picking L random evaluation points a_i

    for (int i = 0; i < L; i++)
        create_zero_matrix<DATA_TYPE>(&query[i], M, K);

    srand(time(NULL));
    for (int i = 0; i < L; i++) {
        alphas[i] = i + 1;
    }


    printf("\n----------- Creating K Random Polynomials -----------\n");
    DATA_TYPE polynomials[K][T + 1]; // Coefficient Matrix

    // Generating random coefficients
    srand(time(NULL));
    for (int i = 0; i < K; i++) {
        polynomials[i][0] = (i == INDEX ? 1 : 0); // Coefficient of X^0 is set to 0/1 based on the chosen INDEX
        for (int j = 1; j < T + 1; j++) {
            polynomials[i][j] = (DATA_TYPE) (rand() % Q);
            if (polynomials[i][j] == 0)
                polynomials[i][j] = 1;
        }
    }

#ifdef DEBUG_MODE
    // Printing the polynomials
    for (int i = 0; i < K; i++) {
        printf("f_%d(x): ", i);
        for (int j = 0; j < T + 1; j++) {
            if (j == 0)
                printf("%f + ", polynomials[i][j]);
            else if (j == T)
                printf("%f x^%d ", polynomials[i][j], j);
            else
                printf("%f x^%d + ", polynomials[i][j], j);
        }
        printf("\n");
    }

    // Printing Alpha Values
    printf("\nAlpha Values: \n");
    for (int i = 0; i < L; i++)
        printf("a_%d: %d \n", i, alphas[i]);
#endif

    // Finishing creating the queries by evalauting all polynomials f_i(x) for a_j for server j
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < K; j++) {
            DATA_TYPE result = 0;
            for (int k = 0; k < T + 1; k++)
                result += polynomials[j][k] * pow(alphas[i], k);
            query[i].data[j] = (DATA_TYPE) ((long) result % Q);
        }
    }


    gettimeofday(&endTime, &tz);
    printf("Query time\n");

    report_running_time(&startTime, &endTime, &tz);
#ifdef DEBUG_MODE
    // Printing Queries
    for (int i = 0; i < L; i++) {
        printf("Query %d [", i);
        for (int j = 0; j < K; j++)
            printf("%f ", query[i].data[j]);
        printf("]\n");
    }
#endif


        printf("\n----------- Sending Query to Server (CPU) to Perform VM Multiplication -----------\n");
        Matrix<DATA_TYPE> response_cpu[L];
        for (int i = 0; i < L; i++) {
            create_zero_matrix<DATA_TYPE>(&response_cpu[i], M, N);
            printf("Sending Query %d \n", i);
            server_cpu<DATA_TYPE>(M, K, N, &query[i], &db, &response_cpu[i], Q);
        }

    #ifdef DEBUG_MODE
        // Printing the Response
        for (int i = 0; i < L; i++)
            print_matrix<DATA_TYPE>(&response_cpu[i], M, N, (char *) "Response");
    #endif


        printf("\n----------- Recovering the Block from the Responses (CPU) Using EasyReecover -----------\n");
        Matrix<long> recovered_block_cpu;
        create_zero_matrix<long>(&recovered_block_cpu, 1, N);



    auto start = std::chrono::high_resolution_clock::now();
//          gettimeofday(&startTime, &tz);

     goldberg_pir_easy_recover(alphas, response_cpu, &recovered_block_cpu, N, L, T, Q);
    auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
std::cout << "duration : " << duration.count() << " microseconds" << std::endl;


//  gettimeofday(&endTime, &tz);


report_running_time(&startTime, &endTime, &tz);


        printf("Comparing the recovered block with the actual database record at index INDEX: ");
        for (int i=0; i < N ;i++){
            if (recovered_block_cpu.data[i] != db.data[INDEX * N  + i]){
                printf("<<Not Equal>>\n\n\n");
                exit(-1);
            }
        }
        printf("<<Equal>>\n\n\n");
}


int main(int argc, char **argv) {
    /* Reading the Query and Database Size (K * N) */
    const int M = atoi(argv[1]); // Query count
    const int K = atoi(argv[2]);
    const int N = atoi(argv[3]);
    const long INDEX = atoi(argv[4]);

    /* Modulus */
    const int L = atoi(argv[5]); // Shamir L Parameter (Total number of Servers)
    const int T = atoi(argv[6]); // Shamir T Parameter
    const int Q = 39551;

    run(M, K, N, INDEX, Q, L, T);

    return 0;
}
