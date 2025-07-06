#include <stdio.h>
#include <sys/time.h>

#define CEIL(M, N) (((M) + (N)-1) / (N))
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/* Macros for changing the datatype from float to double easily in our code */
#define DATA_TYPE float // float for more efficiency > double
#define DATA_TYPE_PACKED float4 // float4 for more efficiency > double4

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

/**
 * \brief GPU-based implementation of Matrix-Matrix multiplication (C_mn = A_mk * B_kn)
 * \param M: Row dimension of A
 * \param K: Column and Row dimension of A and B, respectively
 * \param N: Column dimension of B
 * \param A: Input matrix A
 * \param B: Input matrix B
 * \param C: Output matrix C
 * \param Q: Modulus
 */
template <typename T, const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gpu_matrix_matrix_multiplier(int M, int N, int K,
                                             T *A, T *B,
                                             T *C, int Q = 0) {
    const uint cRow = blockIdx.y;  // output block row
    const uint cCol = blockIdx.x;  // output block column

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    const int extraCols = 5;  // Padding to prevent bank conflicts
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * (BN + extraCols)];

    double threadResults[TM * TN] = {0.0};
    double regM[TM] = {0.0};
    double regN[TN] = {0.0};

    // Global base pointers for this block
    T* A_block = A + cRow * BM * K;
    T* B_block = B + cCol * BN;
    T* C_block = C + cRow * BM * N + cCol * BN;

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // === LOAD A to shared memory (transposed) ===
        uint aRow = threadIdx.x / (BK / 4);
        uint aCol = (threadIdx.x % (BK / 4)) * 4;
        uint globalARow = cRow * BM + aRow;
        uint globalACol = bkIdx + aCol;

        if (globalARow < M && globalACol + 3 < K) {
            float4 tmp = reinterpret_cast<float4*>(&A_block[aRow * K + aCol])[0];
            As[(aCol + 0) * BM + aRow] = tmp.x;
            As[(aCol + 1) * BM + aRow] = tmp.y;
            As[(aCol + 2) * BM + aRow] = tmp.z;
            As[(aCol + 3) * BM + aRow] = tmp.w;
        } else {
            for (int i = 0; i < 4; ++i) {
                if (globalARow < M && (globalACol + i) < K) {
                    As[(aCol + i) * BM + aRow] = static_cast<float>(A_block[aRow * K + aCol + i]);
                } else {
                    As[(aCol + i) * BM + aRow] = 0.0f;
                }
            }
        }

        // === LOAD B to shared memory ===
        uint bRow = threadIdx.x / (BN / 4);
        uint bCol = (threadIdx.x % (BN / 4)) * 4;
        uint globalBRow = bkIdx + bRow;
        uint globalBCol = cCol * BN + bCol;

        if (globalBRow < K && globalBCol + 3 < N) {
            float4 tmp = reinterpret_cast<float4*>(&B_block[bRow * N + bCol])[0];
            Bs[bRow * (BN + extraCols) + bCol + 0] = tmp.x;
            Bs[bRow * (BN + extraCols) + bCol + 1] = tmp.y;
            Bs[bRow * (BN + extraCols) + bCol + 2] = tmp.z;
            Bs[bRow * (BN + extraCols) + bCol + 3] = tmp.w;
        } else {
            for (int i = 0; i < 4; ++i) {
                if (globalBRow < K && (globalBCol + i) < N) {
                    Bs[bRow * (BN + extraCols) + bCol + i] = static_cast<float>(B_block[bRow * N + bCol + i]);
                } else {
                    Bs[bRow * (BN + extraCols) + bCol + i] = 0.0f;
                }
            }
        }

        __syncthreads();

        // === Compute block in double precision ===
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i) {
                regM[i] = static_cast<double>(As[dotIdx * BM + threadRow * TM + i]);
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = static_cast<double>(Bs[dotIdx * (BN + extraCols) + threadCol * TN + i]);
            }

            for (uint i = 0; i < TM; ++i) {
                for (uint j = 0; j < TN; ++j) {
                    threadResults[i * TN + j] += regM[i] * regN[j];
                }
            }
        }

        __syncthreads();

        A_block += BK;
        B_block += BK * N;
    }

    // === Write result to C ===
    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; j += 4) {
            uint row = cRow * BM + threadRow * TM + i;
            uint col = cCol * BN + threadCol * TN + j;

            if (row < M && col + 3 < N) {
                for (int k = 0; k < 4; ++k) {
                    long val = static_cast<long>(C[row * N + col + k]) + static_cast<long>(threadResults[i * TN + j + k]);
                    if (Q)
                        val = (val % Q + Q) % Q;  // Normalize
                    C[row * N + col + k] = static_cast<T>(val);
                }
            } else {
                for (int k = 0; k < 4; ++k) {
                    if (row < M && (col + k) < N) {
                        long val = static_cast<long>(C[row * N + col + k]) + static_cast<long>(threadResults[i * TN + j + k]);
                        if (Q)
                            val = (val % Q + Q) % Q;
                        C[row * N + col + k] = static_cast<T>(val);
                    }
                }
            }
        }
    }
}




/**
 * \brief Runs the Goldberg's PIR server in GPU mode
 * \param M: Row dimension of A
 * \param K: Column and Row dimension of A and B, respectively
 * \param N: Column dimension of B
 * \param A: Pointer to input Matrix A
 * \param B: Pointer to input Matrix B
 * \param C: Pointer to output Matrix C
 * \param Q: Modulus
 */
template<typename T, typename T_PACKED>
void server_gpu(int M, int K, int N, Matrix<T> *A, Matrix<T> *B, Matrix<T> *C, int Q = 0) {
    printf("Sever is running on GPU ...\n");

    /* Allocating Memory for Matrices on the GPU */
    T *A_d, *B_d, *C_d;
    cudaMalloc((void **) &A_d, sizeof(T) * M * K);
    cudaMalloc((void **) &B_d, sizeof(T) * K * N);
    cudaMalloc((void **) &C_d, sizeof(T) * M * N);

    cudaMemcpy(A_d, A->data, sizeof(T) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B->data, sizeof(T) * K * N, cudaMemcpyHostToDevice);

    /* Calculating the time of the GPU computation */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    if (M >= 128 and N >= 128) {
        const uint BM = 128;
        const uint BN = 128;
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim((BM * BN) / (TM * TN));
        gpu_matrix_matrix_multiplier<T, BM, BN, BK, TM, TN>
                <<<gridDim, blockDim>>>(M, N, K, A_d, B_d, C_d, Q);
    } else {
        // this is a hacky solution to the underlying problem
        // of not having proper bounds checking in the kernel
        const uint BM = 64;
        const uint BN = 64;
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim((BM * BN) / (TM * TN));
        gpu_matrix_matrix_multiplier<T, BM, BN, BK, TM, TN>
                <<<gridDim, blockDim>>>(M, N, K, A_d, B_d, C_d, Q);
    }


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Running time for %s: %0.5f (s)\n", "GPU", elapsedTime / 1000);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error : %s\n", cudaGetErrorString(err));
    }

    /* Copying the data back from GPU GMEM to RAM */
    cudaMemcpy(C->data, C_d, sizeof(T) * M * N, cudaMemcpyDeviceToHost);
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

#ifdef DEBUG_MODE
    // Printing Queries
    for (int i = 0; i < L; i++) {
        printf("Query %d [", i);
        for (int j = 0; j < K; j++)
            printf("%f ", query[i].data[j]);
        printf("]\n");
    }
#endif

        printf("\n----------- Sending Query to Server (GPU) to Perform VM Multiplication -----------\n");
        Matrix<DATA_TYPE> response_gpu[L];
        for (int i = 0; i < L; i++) {
            create_zero_matrix<DATA_TYPE>(&response_gpu[i], M, N);
            printf("Sending Query %d \n", i);
            server_gpu<DATA_TYPE, DATA_TYPE_PACKED>(M, K, N, &query[i], &db, &response_gpu[i], Q);
        }

#ifdef DEBUG_MODE
        // Printing the Response
        for (int i = 0; i < L; i++)
            print_matrix<DATA_TYPE>(&response_gpu[i], M, N, (char *) "Response");
#endif

        printf("\n----------- Recovering the Block from the Responses (GPU) Using EasyReecover -----------\n");
        Matrix<long> recovered_block_gpu;
        create_zero_matrix<long>(&recovered_block_gpu, 1, N);

        goldberg_pir_easy_recover(alphas, response_gpu, &recovered_block_gpu, N, L, T, Q);

        printf("Comparing the recovered block with the actual database record at index INDEX: ");
        for (int i = 0; i < N; i++) {
            if (recovered_block_gpu.data[i] != db.data[INDEX * N + i]) {
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
