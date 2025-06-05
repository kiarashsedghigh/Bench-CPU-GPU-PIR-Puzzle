// This file is part of the GPU implementation of the GF2 matrix-vector multiplication
// for the online server cost of the Chor-PIR.
#include "serverComp.cuh"

int main(int argc, char** argv) {

    u32 iterations = 10;
    
    matrix_info_t matrix_info;
    matrix_info_init(&matrix_info, argc, argv);
    iterations = getCmdLineArgumentIntOrDefault(argc, argv, "iterations", iterations);

    matrix_data_t matrix_data;
    matrix_data_init(&matrix_data, &matrix_info);

    dim3 gridDim;
    dim3 blockDim;
    StopWatchInterface* timer = nullptr;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    switch(matrix_info.q > 1) {
        case 0:
            for (u32 algoID = 0 ; algoID < 6; algoID++) {
                sdkResetTimer(&timer);
                CHECK_CUDA(cudaMemset(matrix_data.result, 0, matrix_info.q * matrix_info.b * sizeof(u8)));

                switch (algoID) {
                    case 0:
                        blockDim = dim3(256);
                        gridDim  = dim3((matrix_info.b + 255) / 256);
                        for (u32 i = 0; i < iterations; i++) {
                            sdkStartTimer(&timer);
                            gf2_vector_matrix_mult_basic<<<gridDim, blockDim>>>(matrix_data.rho, matrix_data.DB, matrix_data.result, 
                                matrix_info.r, matrix_info.b);
                            CHECK_CUDA(cudaDeviceSynchronize());
                            sdkStopTimer(&timer);
                        }

                        CHECK_CUDA(cudaMemcpy(matrix_data.h_result, matrix_data.result, matrix_info.b * sizeof(u8), cudaMemcpyDeviceToHost));
                        break;
        
                    case 1:
                        blockDim = dim3(32, 8);
                        gridDim  = dim3((matrix_info.b + 31) / 32);

                        for (u32 i = 0; i < iterations; i++) {
                            sdkStartTimer(&timer);
                            gf2_vector_matrix_mult_optimized_1<<<gridDim, blockDim>>>(matrix_data.rho, matrix_data.DB, matrix_data.result, 
                                matrix_info.r, matrix_info.b);
                            CHECK_CUDA(cudaDeviceSynchronize());
                            sdkStopTimer(&timer);
                        }

                        CHECK_CUDA(cudaMemcpy(matrix_data.h_result, matrix_data.result, matrix_info.b * sizeof(u8), cudaMemcpyDeviceToHost));
                        break;

                    case 2:
                        blockDim = dim3(8, 32);
                        gridDim  = dim3((matrix_info.b + 31) / 32);

                        for (u32 i = 0; i < iterations; i++) {
                            sdkStartTimer(&timer);
                            gf2_vector_matrix_mult_optimized_1_finerSync<<<gridDim, blockDim>>>(matrix_data.rho, matrix_data.DB, matrix_data.result, 
                                matrix_info.r, matrix_info.b);
                            CHECK_CUDA(cudaDeviceSynchronize());
                            sdkStopTimer(&timer);
                        }

                        CHECK_CUDA(cudaMemcpy(matrix_data.h_result, matrix_data.result, matrix_info.b * sizeof(u8), cudaMemcpyDeviceToHost));
                        break;
        
                    case 3:
                        blockDim = dim3(32, 32);
                        gridDim  = dim3(1, ((matrix_info.b/4) + 32) / 32);

                        for (u32 i = 0; i < iterations; i++) {
                            sdkStartTimer(&timer);
                            gf2_vector_matrix_mult_shflxor<<<gridDim, blockDim>>>(matrix_data.rho, matrix_data.DB_32, matrix_data.result_32, 
                                matrix_info.r, matrix_info.b/4);
                            CHECK_CUDA(cudaDeviceSynchronize());
                            sdkStopTimer(&timer);
                        }
                        
                        CHECK_CUDA(cudaMemcpy(matrix_data.h_result, matrix_data.result_32, matrix_info.b * sizeof(u8), cudaMemcpyDeviceToHost));
                        break;
        
                    case 4:
                        for (u32 i = 0; i < iterations; i++) {
                            blockDim = dim3(32, 8);
                            gridDim  = dim3(((matrix_info.b/16) + 31) / 32);
                            sdkStartTimer(&timer);
                            gf2_vector_matrix_mult_optimized_1_uint4<<<gridDim, blockDim>>>(matrix_data.rho, matrix_data.DB_32_4, matrix_data.result_32_4, 
                                matrix_info.r, matrix_info.b/16);
                            CHECK_CUDA(cudaDeviceSynchronize());
                            sdkStopTimer(&timer);
                        }

                        CHECK_CUDA(cudaMemcpy(matrix_data.h_result, matrix_data.result_32_4, matrix_info.b * sizeof(u8), cudaMemcpyDeviceToHost));
                        break;
                                
                    case 5:
                        for (u32 i = 0; i < iterations; i++) {
                            blockDim = dim3(32, 8);
                            gridDim  = dim3(((matrix_info.b/32) + 31) / 32);
                            sdkStartTimer(&timer);
                            gf2_vector_matrix_mult_optimized_1_ulong4<<<gridDim, blockDim>>>(matrix_data.rho, matrix_data.DB_64_4, matrix_data.result_64_4, 
                                matrix_info.r, matrix_info.b/32);
                            CHECK_CUDA(cudaDeviceSynchronize());
                            sdkStopTimer(&timer);
                        }

                        CHECK_CUDA(cudaMemcpy(matrix_data.h_result, matrix_data.result_64_4, matrix_info.b * sizeof(u8), cudaMemcpyDeviceToHost));
                        break;
                }

                printf("ALG %d: ", algoID);
                printf("\tRUNTIME: %5.3f ms\n", sdkGetAverageTimerValue(&timer));
                // printf("\tOUT: ");
                // for (int i = 0; i < 64; i++) {
                //     printf("%u", matrix_data.h_result[i]);
                // }
                // printf("\n");

                // printf("\tOUT: ");
                // for (int i = 64; i < 128; i++) {
                //     printf("%u", matrix_data.h_result[i]);
                // }
                // printf("\n");
            }
            break;

        case 1:
            // batch mode
            for (u32 algoID = 0 ; algoID < 4; algoID++) {
                sdkResetTimer(&timer);
                CHECK_CUDA(cudaMemset(matrix_data.result, 0, matrix_info.q * matrix_info.b * sizeof(u8)));

                switch (algoID) {
                    case 0:
                        for (u32 i = 0; i < iterations; i++) {
                            blockDim = dim3(32, 8);
                            gridDim  = dim3((matrix_info.b + 31) / 32, matrix_info.q);
                            sdkStartTimer(&timer);
                            gf2_vector_matrix_mult_batch<<<gridDim, blockDim>>>(matrix_data.rho, matrix_data.DB, matrix_data.result, 
                                matrix_info.r, matrix_info.b, matrix_info.q);
                            CHECK_CUDA(cudaDeviceSynchronize());
                            sdkStopTimer(&timer);
                            break;
                        }

                        CHECK_CUDA(cudaMemcpy(matrix_data.h_result, matrix_data.result, matrix_info.b * matrix_info.q * sizeof(u8), cudaMemcpyDeviceToHost));
                        break;

                    case 1:
                        for (u32 i = 0; i < iterations; i++) {
                            blockDim = dim3(32, 8);
                            gridDim  = dim3(((matrix_info.b / 4) + 31) / 32, matrix_info.q);
                            sdkStartTimer(&timer);
                            gf2_vector_matrix_mult_batch_u32<<<gridDim, blockDim>>>(matrix_data.rho, matrix_data.DB_32, matrix_data.result_32, 
                                matrix_info.r, matrix_info.b / 4, matrix_info.q);
                            CHECK_CUDA(cudaDeviceSynchronize());
                            sdkStopTimer(&timer);
                            break;
                        }

                        CHECK_CUDA(cudaMemcpy(matrix_data.h_result, matrix_data.result_32, matrix_info.b * matrix_info.q * sizeof(u8), cudaMemcpyDeviceToHost));
                        break;

                    case 2:
                        for (u32 i = 0; i < iterations; i++) {
                            blockDim = dim3(32, 8);
                            gridDim  = dim3(((matrix_info.b / 16) + 31) / 32, matrix_info.q);
                            sdkStartTimer(&timer);
                            gf2_vector_matrix_mult_batch_uint4<<<gridDim, blockDim>>>(matrix_data.rho, matrix_data.DB_32_4, matrix_data.result_32_4, 
                                matrix_info.r, matrix_info.b / 16, matrix_info.q);
                            CHECK_CUDA(cudaDeviceSynchronize());
                            sdkStopTimer(&timer);
                            break;
                        }

                        CHECK_CUDA(cudaMemcpy(matrix_data.h_result, matrix_data.result_32_4, matrix_info.b * matrix_info.q * sizeof(u8), cudaMemcpyDeviceToHost));
                        break;

                    case 3:
                        for (u32 i = 0; i < iterations; i++) {
                            blockDim = dim3(32, 8);
                            gridDim  = dim3(((matrix_info.b / 32) + 31) / 32, matrix_info.q);
                            sdkStartTimer(&timer);
                            gf2_vector_matrix_mult_batch_ulong4<<<gridDim, blockDim>>>(matrix_data.rho, matrix_data.DB_64_4, matrix_data.result_64_4, 
                                matrix_info.r, matrix_info.b / 32, matrix_info.q);
                            CHECK_CUDA(cudaDeviceSynchronize());
                            sdkStopTimer(&timer);
                            break;
                        }

                        CHECK_CUDA(cudaMemcpy(matrix_data.h_result, matrix_data.result_64_4, matrix_info.b * matrix_info.q * sizeof(u8), cudaMemcpyDeviceToHost));
                        break;
                }

                if (algoID==2) {
                    // Save runtime and algo ID to file
                    FILE* fp = fopen("runtime_results.txt", "a");
                    if (fp) {
                        fprintf(fp, "q::%10d:: b::%10d:: r::%10d ::RUNTIME:: %5.3f ms\n", matrix_info.q, matrix_info.b, matrix_info.r, sdkGetAverageTimerValue(&timer));
                        fclose(fp);
                    } else {
                        printf("Error opening file for writing results.\n");
                    }
                }
                printf("ALG %d: ", algoID);
                printf("\tRUNTIME: %5.3f ms\n", sdkGetAverageTimerValue(&timer));
                // printf("\tOUT: ");
                // for (int i = 0; i < 64; i++) {
                //     printf("%u", matrix_data.h_result[i]);
                // }
                // printf("\n");

                // printf("\tOUT: ");
                // for (int i = 64; i < 128; i++) {
                //     printf("%u", matrix_data.h_result[i]);
                // }
                // printf("\n");
            }
    }
    
    sdkDeleteTimer(&timer);
    matrix_data_free(&matrix_data);
    
    return 0;
}
