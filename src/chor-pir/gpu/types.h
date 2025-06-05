#ifndef TYPES_H
#define TYPES_H

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

typedef struct {
    u32 r; // number of rows
    u32 b; // number of columns
    u32 q; // number of queries
    u32 qmax; // maximum number of queries
} matrix_info_t;

typedef struct {
    u8* rho;      // Pointer to the rho vector
    u8* DB;       // Pointer to the database matrix
    u32 *DB_32;
    uint4 *DB_32_4;
    ulong4 *DB_64_4;
    u8* result;   // Pointer to the result vector
    u32 *result_32;
    uint4 *result_32_4;
    ulong4 *result_64_4;
    u8* h_result; // Host pointer to the result vector
    matrix_info_t info; // Matrix information (rows, columns, queries)
} matrix_data_t;


#endif // TYPES_H