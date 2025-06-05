
#include "util.cuh"


u32 matrix_info_init(matrix_info_t* info, int argc, char** argv) {
    if (info == NULL) {
        fprintf(stderr, "Error: matrix_info_t pointer is NULL.\n");
        return 0;
    }
    
    info->qmax = 1 << 10;   // maximum number of queries
    info->r = 1 << (getCmdLineArgumentIntOrDefault(argc, argv, "r", 17));
    info->b = 1 << (getCmdLineArgumentIntOrDefault(argc, argv, "b", 12));
    info->q = 1 << (getCmdLineArgumentIntOrDefault(argc, argv, "q", 1 << 5));

    return 1; // success
}

u32 matrix_data_init(matrix_data_t* data, matrix_info_t* info) {
    if (data == NULL || info == NULL) {
        fprintf(stderr, "Error: matrix_data_t or matrix_info_t pointer is NULL.\n");
        return 0;
    }

    data->info.r = info->r;
    data->info.b = info->b;
    data->info.q = info->q;
    data->info.qmax = info->qmax;

    size_t rho_size = data->info.q * data->info.r * sizeof(u8);
    size_t db_size = data->info.r * data->info.b * sizeof(u8);
    CHECK_CUDA(cudaMalloc((void**)&data->rho, rho_size));
    CHECK_CUDA(cudaMalloc((void**)&data->DB, db_size));
    CHECK_CUDA(cudaMalloc((void**)&data->result, data->info.q * data->info.b * sizeof(u8)));

    u8* h_DB  = (u8*) malloc(db_size);
    u8* h_rho = (u8*) malloc(rho_size);
    data->h_result = (u8*) malloc(data->info.q * data->info.b * sizeof(u8));
    if (h_DB == NULL || h_rho == NULL) {
        fprintf(stderr, "Error: Failed to allocate host memory for the matrix or rho vector.\n");
        return 0;
    }

    FILE* db_file = fopen("db.bin", "rb");
    if (db_file) {
        size_t read_bytes = fread(h_DB, sizeof(u8), db_size, db_file);
        if (read_bytes != db_size) {
            fprintf(stderr, "Error: Could not read full DB from file.\n");
            fclose(db_file);
            exit(EXIT_FAILURE);
        }
        fclose(db_file);
    } else {
        fprintf(stderr, "Error: Could not open db.bin file.\n");
        exit(EXIT_FAILURE);
    }

    // Load h_rho from file
    FILE* rho_file = fopen("rho.bin", "rb");
    if (rho_file) {
        size_t read_bytes = fread(h_rho, sizeof(u8), rho_size, rho_file);
        if (read_bytes != rho_size) {
            fprintf(stderr, "Error: Could not read full DB from file.\n");
            fclose(rho_file);
            exit(EXIT_FAILURE);
        }
        fclose(rho_file);
    } else {
        fprintf(stderr, "Error: Could not open db.bin file.\n");
        exit(EXIT_FAILURE);
    }
    
    CHECK_CUDA(cudaMemcpy(data->DB, h_DB, db_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(data->rho, h_rho, rho_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(data->result, 0, data->info.q * data->info.b * sizeof(u8)));
    data->DB_32 = (u32*) data->DB;
    data->DB_32_4 = (uint4*) data->DB;
    data->DB_64_4 = (ulong4*) data->DB;
    data->result_32 = (u32*) data->result;
    data->result_32_4 = (uint4*) data->result;
    data->result_64_4 = (ulong4*) data->result;
    
    free(h_DB);
    free(h_rho);
    
    return 1;
}

u32 matrix_data_free(matrix_data_t* data) {
    if (data == NULL) {
        fprintf(stderr, "Error: matrix_data_t pointer is NULL.\n");
        return 0;
    }

    // Free the allocated memory for the matrix
    CHECK_CUDA(cudaFree(data->DB));
    CHECK_CUDA(cudaFree(data->rho));
    CHECK_CUDA(cudaFree(data->result));
    free(data->h_result);
    
    return 1; // success
}
