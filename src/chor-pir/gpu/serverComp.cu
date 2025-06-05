
#include "serverComp.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void gf2_vector_matrix_mult_basic(
    const u8* __restrict__ rho,    // [r] bytes
    const u8* __restrict__ DB,     // [r][b] bytes
    u8* result,                    // [b] bytes
    u32 r,
    u32 b)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    u8 acc = 0;
    for (u32 i = 0; i < r; i++) {
            acc ^= (rho[i] == 1) ? DB[i * b + col] : 0;
    }
    result[col] = acc;
}

__global__ void gf2_vector_matrix_mult_optimized_1(
    const u8* __restrict__ rho,    // [r] bytes
    const u8* __restrict__ DB,     // [r][b] bytes
    u8* result,                    // [b] bytes
    u32 r,
    u32 b)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // if (col >= b) return;

    u8 acc = 0;
    for (int i = threadIdx.y; i < r; i += blockDim.y) {
        u8 mask = -(rho[i] & 1);  // 0 or 0xFF
        acc ^= DB[i * b + col] & mask;
    }

    // Store each thread's accumulator in shared memory
    __shared__ u8 smem[32][32]; // Supports up to 32x32 blockDim
    smem[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();

    if (threadIdx.y == 0) {
        u8 final = 0;
        for (u32 k = 0; k < blockDim.y; ++k)
            final ^= smem[k][threadIdx.x];
        result[col] = final;
    }
}

__global__ void gf2_vector_matrix_mult_optimized_1_finerSync(
    const u8* __restrict__ rho,    // [r] bytes
    const u8* __restrict__ DB,     // [r][b] bytes
    u8* result,                    // [b] bytes
    u32 r,
    u32 b)
{
    u32 col = blockIdx.x * blockDim.y + threadIdx.y;

    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<8> tile = cg::tiled_partition<8>(cta);

    u8 acc = 0;

    // Each thread (within its warp) handles different i values
    for (u32 i = threadIdx.x; i < r; i += blockDim.x) {
        u8 mask = -(rho[i] & 1);  // 0 or 0xFF
        acc ^= DB[i * b + col] & mask;
    }

    // Shared memory for intra-warp reduction
    __shared__ u8 smem[32][32];  // [threadIdx.x][threadIdx.y]
    smem[threadIdx.x][threadIdx.y] = acc;

    // Synchronize within each warp (32 threads with same y)
    tile.sync();

    if (tile.thread_rank() == 0) {
        u8 final = 0;
        for (int k = 0; k < 8; ++k)
            final ^= smem[k][threadIdx.y];
        result[col] = final;
    }
}

__global__ void gf2_vector_matrix_mult_shflxor(
    const u8* __restrict__ rho,
    const u32* __restrict__ DB,
    u32* result,
    u32 r,
    u32 b)
{
    u32 row = blockIdx.y * blockDim.y + threadIdx.y;
    // if (row >= b) return;

    u32 acc = 0;
    
    for (u32 i = threadIdx.x; i < r; i += 32) {
        u32 mask = -(rho[i] & 1);
        acc ^= DB[i * b + row] & mask;
    }

    u32 lane_mask = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset /= 2)
        acc ^= __shfl_xor_sync(lane_mask, acc, offset);

    if (threadIdx.x == 0) {
        result[row] = (u32)acc;
    }
}

__global__ void gf2_vector_matrix_mult_optimized_1_uint4(
    const u8* __restrict__ rho,    // [r] bytes
    const uint4* __restrict__ DB,     // [r][b] bytes
    uint4* result,                    // [b] bytes
    u32 r,
    u32 b)
{
    u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    // if (col >= b) return;

    uint4 acc = {0, 0, 0, 0};
    for (u32 i = threadIdx.y; i < r; i += blockDim.y) {
        // u32 mask = -(rho[i] & 1);  // 0 or 0xFF
        // uint4 val = DB[i * b + col];
        // acc.x ^= (val.x & mask);
        // acc.y ^= (val.y & mask);
        // acc.z ^= (val.z & mask);
        // acc.w ^= (val.w & mask);

        u8 rho_bit = __ldg(&rho[i]) & 1;
        if (rho_bit) {
            uint4 val = DB[i * b + col];
            acc.x ^= val.x;
            acc.y ^= val.y;
            acc.z ^= val.z;
            acc.w ^= val.w;
        }
    }

    // Store each thread's accumulator in shared memory
    __shared__ uint4 smem[8][32]; // Supports up to 32x32 blockDim
    smem[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();

    if (threadIdx.y == 0) {
        uint4 final = {0, 0, 0, 0};
        for (int k = 0; k < blockDim.y; ++k) {
            final.x ^= smem[k][threadIdx.x].x;
            final.y ^= smem[k][threadIdx.x].y;
            final.z ^= smem[k][threadIdx.x].z;
            final.w ^= smem[k][threadIdx.x].w;
        }
        result[col] = final;
    }
}

__global__ void gf2_vector_matrix_mult_optimized_1_ulong4(
    const u8* __restrict__ rho,       // [r] bytes
    const ulong4* __restrict__ DB,     // [r][b] bytes
    ulong4* result,                    // [b] bytes
    u32 r,
    u32 b)
{
    u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    // if (col >= b) return;

    ulong4 acc = {0, 0, 0, 0};
    for (u32 i = threadIdx.y; i < r; i += blockDim.y) {
        ulong mask = -(rho[i] & 1);  // 0 or 0xFF
        ulong4 val = DB[i * b + col];
        acc.x ^= (val.x & mask);
        acc.y ^= (val.y & mask);
        acc.z ^= (val.z & mask);
        acc.w ^= (val.w & mask);
    }

    __shared__ ulong4 smem[32][32];
    smem[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();

    if (threadIdx.y == 0) {
        ulong4 final = {0, 0, 0, 0};
        for (int k = 0; k < blockDim.y; ++k) {
            final.x ^= smem[k][threadIdx.x].x;
            final.y ^= smem[k][threadIdx.x].y;
            final.z ^= smem[k][threadIdx.x].z;
            final.w ^= smem[k][threadIdx.x].w;
        }
        result[col] = final;
    }
}

__global__ void gf2_vector_matrix_mult_optimized_2(
    const u8* __restrict__ rho,
    const u8* __restrict__ DB,
    u8* result,
    u32 r,
    u32 b)
{
    const u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 tid_y = threadIdx.y;

    __shared__ u8 tile_rho[32];               // blockDim.y
    __shared__ u8 tile_DB[32][32];            // [rows][cols]

    u8 acc = 0;

    for (u32 i_base = 0; i_base < r; i_base += blockDim.y) {
        u32 row = i_base + tid_y;

        // Load rho and DB into shared memory
        if (threadIdx.x == 0 && row < r)
            tile_rho[tid_y] = rho[row];

        if (row < r && col < b)
            tile_DB[tid_y][threadIdx.x] = DB[row * b + col];

        __syncthreads();

        // Compute XOR locally
        for (u32 k = 0; k < blockDim.y && (i_base + k) < r; ++k) {
            u8 mask = -(tile_rho[k] & 1);
            acc ^= tile_DB[k][threadIdx.x] & mask;
        }

        __syncthreads(); // reset for next tile
    }

    __shared__ u8 smem[32][32];
    smem[tid_y][threadIdx.x] = acc;
    __syncthreads();

    if (tid_y == 0) {
        u8 final = 0;
        for (u32 k = 0; k < blockDim.y; ++k)
            final ^= smem[k][threadIdx.x];
        result[col] = final;
    }
}

__global__ void gf2_vector_matrix_mult_batch(
    const u8* __restrict__ rho,     // [q][r]
    const u8* __restrict__ DB,      // [r][b]
    u8* result,                     // [q][b]
    u32 r,
    u32 b,
    u32 q) {
    const u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 query   = blockIdx.y;

    // if (col >= b || query >= q) return;

    u8 acc = 0;
    for (u32 i = threadIdx.y; i < r; i += blockDim.y) {
        u8 mask = -(rho[query * r + i] & 1); // rho[query][i]
        acc ^= DB[i * b + col] & mask;
    }

    // Warp-wide reduction
    __shared__ u8 smem[32][32];  // [threadIdx.y][threadIdx.x]
    smem[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();

    if (threadIdx.y == 0) {
        u8 final = 0;
        for (u32 k = 0; k < blockDim.y; ++k)
            final ^= smem[k][threadIdx.x];
        result[query * b + col] = final;
    }
}

__global__ void gf2_vector_matrix_mult_batch_u32(
    const u8* __restrict__ rho,      // [q][r]
    const u32* __restrict__ DB,      // [r][b]
    u32* result,                     // [q][b]
    u32 r,
    u32 b,
    u32 q) {
    const u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 query   = blockIdx.y;

    // if (col >= b || query >= q) return;

    u32 acc = 0;
    for (u32 i = threadIdx.y; i < r; i += blockDim.y) {
        u32 mask = -(rho[query * r + i] & 1); // rho[query][i]
        acc ^= DB[i * b + col] & mask;
    }

    // Warp-wide reduction
    __shared__ u32 smem[32][32];  // [threadIdx.y][threadIdx.x]
    smem[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();

    if (threadIdx.y == 0) {
        u32 final = 0;
        for (u32 k = 0; k < blockDim.y; ++k)
            final ^= smem[k][threadIdx.x];
        result[query * b + col] = final;
    }
}

__global__ void gf2_vector_matrix_mult_batch_uint4(
    const u8* __restrict__ rho,        // [q][r]
    const uint4* __restrict__ DB,      // [r][b]
    uint4* result,                     // [q][b]
    u32 r,
    u32 b,
    u32 q) 
{
    const u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 query   = blockIdx.y;
    // if (col >= b || query >= q) return;

    uint4 acc = {0, 0, 0, 0};
    #pragma unroll 4
    for (u32 i = threadIdx.y; i < r; i += blockDim.y) {
        // u8 rho_shared = __ldg(&rho[query * r + i]);
        // const uint4 val = __ldg(&DB[i * b + col]);
        // // __syncthreads();
        
        // u32 mask = -(rho_shared & 1); // rho[query][i]
        // acc.x ^= val.x & mask;
        // acc.y ^= val.y & mask;
        // acc.z ^= val.z & mask;
        // acc.w ^= val.w & mask;

        u8 rho_bit = __ldg(&rho[query * r + i]) & 1;
        if (rho_bit) {
            const uint4 val = __ldg(&DB[i * b + col]);
            acc.x ^= val.x;
            acc.y ^= val.y;
            acc.z ^= val.z;
            acc.w ^= val.w;
        }
    }

    // Warp-wide reduction
    __shared__ uint4 smem[8][32];  // [threadIdx.y][threadIdx.x]
    smem[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();

    if (threadIdx.y == 0) {
        uint4 final = {0, 0, 0, 0};
        #pragma unroll
        for (u32 k = 0; k < blockDim.y; ++k) {
            final.x ^= smem[k][threadIdx.x].x;
            final.y ^= smem[k][threadIdx.x].y;
            final.z ^= smem[k][threadIdx.x].z;
            final.w ^= smem[k][threadIdx.x].w;
        }
        result[query * b + col] = final;
    }
}

__global__ void gf2_vector_matrix_mult_batch_uint4_shflxor(
    const u8* __restrict__ rho,        // [q][r]
    const uint4* __restrict__ DB,      // [r][b]
    uint4* result,                     // [q][b]
    u32 r,
    u32 b,
    u32 q
) {
    const u32 col   = blockIdx.x;       // each warp handles 1 column of uint4
    const u32 query = blockIdx.y;       // batch index
    const u32 lane  = threadIdx.x;      // lane index in warp

    // if (col >= b / 4 || query >= q) return;

    uint4 acc = {0, 0, 0, 0};

    for (u32 i = lane; i < r; i += 32) {
        u32 mask = -(rho[query * r + i] & 1);
        const uint4 val = DB[i * (b / 4) + col];  // row-major access
        acc.x ^= val.x & mask;
        acc.y ^= val.y & mask;
        acc.z ^= val.z & mask;
        acc.w ^= val.w & mask;
    }

    // Warp-wide XOR reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc.x ^= __shfl_xor_sync(0xFFFFFFFF, acc.x, offset);
        acc.y ^= __shfl_xor_sync(0xFFFFFFFF, acc.y, offset);
        acc.z ^= __shfl_xor_sync(0xFFFFFFFF, acc.z, offset);
        acc.w ^= __shfl_xor_sync(0xFFFFFFFF, acc.w, offset);
    }

    if (lane == 0) {
        result[query * (b / 4) + col] = acc;
    }
}

__global__ void gf2_vector_matrix_mult_batch_ulong4(
    const u8* __restrict__ rho,      // [q][r]
    const ulong4* __restrict__ DB,      // [r][b]
    ulong4* result,                     // [q][b]
    u32 r,
    u32 b,
    u32 q) {
    const u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 query   = blockIdx.y;

    // if (col >= b || query >= q) return;

    ulong4 acc = {0, 0, 0, 0};
    for (u32 i = threadIdx.y; i < r; i += blockDim.y) {
        ulong mask = -(rho[query * r + i] & 1); // rho[query][i]
        acc.x ^= DB[i * b + col].x & mask;
        acc.y ^= DB[i * b + col].y & mask;
        acc.z ^= DB[i * b + col].z & mask;
        acc.w ^= DB[i * b + col].w & mask;
    }

    // Warp-wide reduction
    __shared__ ulong4 smem[32][32];  // [threadIdx.y][threadIdx.x]
    smem[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();

    if (threadIdx.y == 0) {
        ulong4 final = {0, 0, 0, 0};
        for (u32 k = 0; k < blockDim.y; ++k) {
            final.x ^= smem[k][threadIdx.x].x;
            final.y ^= smem[k][threadIdx.x].y;
            final.z ^= smem[k][threadIdx.x].z;
            final.w ^= smem[k][threadIdx.x].w;
        }
        result[query * b + col] = final;
    }
}

//