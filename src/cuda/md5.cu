#include <stdio.h>
#include <cuda_runtime.h>
#include <limits.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#define A0 (0x67452301)
#define B0 (0xefcdab89)
#define C0 (0x98badcfe)
#define D0 (0x10325476)
#define DIGEST_SIZE (16)
#define CHUNK_SIZE (64)
#define WORD_SIZE (4)
// How many MD5 hashes do we want to compute concurrently?
#define BATCH_SIZE (16384)
#define CEIL(x) ((x) == (int)(x) ? (int)(x) : ((x) > 0 ? (int)(x) + 1 : (int)(x)))

typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef unsigned long long uint64_t;

// Shift amounts in each MD5 round
__constant__ uint32_t shift_amts[64] = {
    7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
    5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
    4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
    6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,
};
// Integer parts of signs of integers; used during the MD5 round operations
__constant__ uint32_t k_table[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
};
// Context of MD5 computation
struct md5_ctx {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
};
// A vector of bytes; used to interface with the Rust code
struct FfiVector {
    uint8_t *data;
    size_t len;
};
// A vector of FfiVectors; used to interface with the Rust code
struct FfiVectorBatch {
    FfiVector *data;
    size_t len;
};

// Repeat initial context BATCH_SIZE times
md5_ctx init_ctxs[BATCH_SIZE];

// Left rotate 32-bit integer x by amt
__device__ uint32_t leftrotate(uint32_t x, uint32_t amt) {
    return (x << (amt % 32)) | (x >> (32 - (amt % 32)));
}

// Preprocess a batch of messages
__global__ void md5_preprocess_batched(uint8_t *pre_processed_msgs, size_t *pre_processed_sizes, size_t *orig_sizes, size_t *culmn_sizes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Bounds-checking
    if (idx < BATCH_SIZE) {
        size_t n = orig_sizes[idx];
        uint64_t size_in_bits = 8ULL * n;
        size_t pre_processed_size = pre_processed_sizes[idx];

        // Add 0x80 byte
        pre_processed_msgs[culmn_sizes[idx] + n] = 0x80;
        // Adding the length
        for (size_t i = pre_processed_size - 8; i < pre_processed_size; i++) {
            size_t offset = i - (pre_processed_size - 8);
            pre_processed_msgs[culmn_sizes[idx] + (pre_processed_size - 8) + ((pre_processed_size - i) - 1)] =
                (size_in_bits >> ((7 - offset) * 8)) & 0xff;
        }
    }
}

// Modifying the contexts; this is the core of the MD5 computation
__global__ void md5_compute_batched(md5_ctx *ctxs, uint8_t *pre_processed_msgs, size_t *pre_processed_sizes, size_t *culmn_sizes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Bounds-checking
    if (idx < BATCH_SIZE) {
        size_t pre_processed_size = pre_processed_sizes[idx];
        // Index using culminative sizes
        uint8_t *pre_processed_msg = pre_processed_msgs + culmn_sizes[idx];
        md5_ctx ctx = ctxs[idx];

        // Iterate over 64-byte chunks of the pre-processed message
        for (uint8_t *chunk = pre_processed_msg; chunk < pre_processed_msg + pre_processed_size; chunk += CHUNK_SIZE) {
            uint32_t words[CHUNK_SIZE / WORD_SIZE] = {0};

            // Break up the current chunk into words
            for (int word_idx = 0; word_idx < CHUNK_SIZE; word_idx += WORD_SIZE) {
                words[word_idx / WORD_SIZE] = chunk[word_idx] +
                                              (chunk[word_idx + 1] << 8) +
                                              (chunk[word_idx + 2] << 16) +
                                              (chunk[word_idx + 3] << 24);
            }

            // Start round
            uint32_t a = ctx.a;
            uint32_t b = ctx.b;
            uint32_t c = ctx.c;
            uint32_t d = ctx.d;

            // 64 round operations
            for (int i = 0; i < CHUNK_SIZE; i++) {
                uint32_t f;
                uint32_t g;

                if (i <= 15) {
                    f = ((b & c) | ((~b) & d));
                    g = i;
                } else if (16 <= i && i <= 31) {
                    f = ((d & b) | ((~d) & c));
                    g = (5*i + 1) % 16;
                } else if (32 <= i && i <= 47) {
                    f = (b ^ c ^ d) ;
                    g = (3*i + 5) % 16;
                } else {
                    f = (c ^ (b | (~d)))  ;
                    g = (7*i) % 16;
                }

                f = (f + a + k_table[i] + words[g]);
                a = d;
                d = c;
                c = b;
                b = (b + leftrotate(f, shift_amts[i]));
            }

            // Add to current registers
            ctxs[idx].a = (ctxs[idx].a + a);
            ctxs[idx].b = (ctxs[idx].b + b);
            ctxs[idx].c = (ctxs[idx].c + c);
            ctxs[idx].d = (ctxs[idx].d + d);
        }
    }
}

// Each thread compares its respective context from ctxs with the target_ctx
// and writes its index on the grid to match_idx if the contexts are identical
__global__ void md5_compare_ctx_batched(md5_ctx *ctxs, md5_ctx *target_ctx, int *match_idx) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < BATCH_SIZE) {
        md5_ctx ctx = ctxs[idx];

        if (ctx.a == target_ctx->a &&
            ctx.b == target_ctx->b &&
            ctx.c == target_ctx->c &&
            ctx.d == target_ctx->d) {
                *match_idx = idx;
            }
    }
}

// Attempt to find a message in the batch whose digest matches the target digest
int md5_target_batched(FfiVector *msgs, md5_ctx *h_target_ctx) {
    uint8_t *h_pre_processed_msgs;
    uint8_t *d_pre_processed_msgs;
    size_t h_pre_processed_sizes[BATCH_SIZE];
    size_t *d_pre_processed_sizes;
    size_t h_orig_sizes[BATCH_SIZE];
    size_t *d_orig_sizes;
    size_t h_culmn_sizes[BATCH_SIZE];
    size_t *d_culmn_sizes;
    md5_ctx *d_ctxs;
    md5_ctx *d_target_ctx;
    int *d_match_idx;
    int h_match_idx = -1;
    size_t total_size = 0;
    const size_t max_total_size = 128 * 1024 * 1024;
    const int threads_per_block = 32;
    const int blocks_per_grid = CEIL((float)BATCH_SIZE / (float)threads_per_block);

    // Calculate the total size of the messages after pre-processing
    // We also fill in the size arrays, such as the size of each message after pre-processing
    for (int i = 0; i < BATCH_SIZE; i++) {
        // Size of the i-th message after pre-processing
        // It is calculated as such because we have to fit in the message (msgs[i].len bytes), the message's length
        // as a 64-bit integer (8 bytes), and the additional 0x80 byte (1 byte)
        size_t padded = msgs[i].len + 8 + 1 + (CHUNK_SIZE - 1);
        size_t pre_processed_size = (padded / CHUNK_SIZE) * CHUNK_SIZE;
        if (pre_processed_size == 0 || total_size > SIZE_MAX - pre_processed_size) {
            return -1;
        }
        h_pre_processed_sizes[i] = pre_processed_size;
        h_orig_sizes[i] = msgs[i].len;
        h_culmn_sizes[i] = (i == 0 ? 0 : h_culmn_sizes[i - 1] + pre_processed_size);
        total_size += pre_processed_size;
        if (total_size > max_total_size) {
            return -1;
        }
    }

    // Allocate enough memory for all of the pre-processed messages
    h_pre_processed_msgs = new uint8_t[total_size];
    // Memzeroing it eliminates the need for zero padding
    memset(h_pre_processed_msgs, 0, total_size);

    // Memcpy each message to its corresponding index
    for (int i = 0; i < BATCH_SIZE; i++) {
        if (h_culmn_sizes[i] + msgs[i].len > total_size) {
            delete[] h_pre_processed_msgs;
            return -1;
        }
        memcpy(h_pre_processed_msgs + h_culmn_sizes[i], msgs[i].data, msgs[i].len);
    }
    // Allocate space for the pre-processed messages on the device
    if (cudaMalloc(&d_pre_processed_msgs, total_size) != cudaSuccess) {
        delete[] h_pre_processed_msgs;
        return -1;
    }
    // Array of culminative message sizes
    if (cudaMalloc(&d_culmn_sizes, sizeof(size_t) * BATCH_SIZE) != cudaSuccess) {
        cudaFree(d_pre_processed_msgs);
        delete[] h_pre_processed_msgs;
        return -1;
    }
    // Array of pre-processed message sizes
    if (cudaMalloc(&d_pre_processed_sizes, sizeof(size_t) * BATCH_SIZE) != cudaSuccess) {
        cudaFree(d_culmn_sizes);
        cudaFree(d_pre_processed_msgs);
        delete[] h_pre_processed_msgs;
        return -1;
    }
    // Array of original message sizes
    if (cudaMalloc(&d_orig_sizes, sizeof(size_t) * BATCH_SIZE) != cudaSuccess) {
        cudaFree(d_pre_processed_sizes);
        cudaFree(d_culmn_sizes);
        cudaFree(d_pre_processed_msgs);
        delete[] h_pre_processed_msgs;
        return -1;
    }
    // Array of MD5 contexts
    if (cudaMalloc(&d_ctxs, sizeof(md5_ctx) * BATCH_SIZE) != cudaSuccess) {
        cudaFree(d_orig_sizes);
        cudaFree(d_pre_processed_sizes);
        cudaFree(d_culmn_sizes);
        cudaFree(d_pre_processed_msgs);
        delete[] h_pre_processed_msgs;
        return -1;
    }
    // The target context
    if (cudaMalloc(&d_target_ctx, sizeof(md5_ctx)) != cudaSuccess) {
        cudaFree(d_ctxs);
        cudaFree(d_orig_sizes);
        cudaFree(d_pre_processed_sizes);
        cudaFree(d_culmn_sizes);
        cudaFree(d_pre_processed_msgs);
        delete[] h_pre_processed_msgs;
        return -1;
    }
    // The integer threads write a match to (if one is found)
    if (cudaMalloc(&d_match_idx, sizeof(int)) != cudaSuccess) {
        cudaFree(d_target_ctx);
        cudaFree(d_ctxs);
        cudaFree(d_orig_sizes);
        cudaFree(d_pre_processed_sizes);
        cudaFree(d_culmn_sizes);
        cudaFree(d_pre_processed_msgs);
        delete[] h_pre_processed_msgs;
        return -1;
    }
    //  Memcpys to the variables allocated above
    if (cudaMemcpy(d_pre_processed_msgs, h_pre_processed_msgs, total_size, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_culmn_sizes, h_culmn_sizes, sizeof(size_t) * BATCH_SIZE, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_pre_processed_sizes, h_pre_processed_sizes, sizeof(size_t) * BATCH_SIZE, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_orig_sizes, h_orig_sizes, sizeof(size_t) * BATCH_SIZE, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_ctxs, &init_ctxs, sizeof(md5_ctx) * BATCH_SIZE, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_target_ctx, h_target_ctx, sizeof(md5_ctx), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_match_idx, &h_match_idx, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_match_idx);
        cudaFree(d_target_ctx);
        cudaFree(d_ctxs);
        cudaFree(d_orig_sizes);
        cudaFree(d_pre_processed_sizes);
        cudaFree(d_culmn_sizes);
        cudaFree(d_pre_processed_msgs);
        delete[] h_pre_processed_msgs;
        return -1;
    }

    // Preprocess the messages
    md5_preprocess_batched<<<blocks_per_grid, threads_per_block>>>(d_pre_processed_msgs, d_pre_processed_sizes, d_orig_sizes, d_culmn_sizes);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_match_idx);
        cudaFree(d_target_ctx);
        cudaFree(d_ctxs);
        cudaFree(d_orig_sizes);
        cudaFree(d_pre_processed_sizes);
        cudaFree(d_culmn_sizes);
        cudaFree(d_pre_processed_msgs);
        delete[] h_pre_processed_msgs;
        return -1;
    }
    // Modify the states (the core of the MD5 computation)
    md5_compute_batched<<<blocks_per_grid, threads_per_block>>>(d_ctxs, d_pre_processed_msgs, d_pre_processed_sizes, d_culmn_sizes);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_match_idx);
        cudaFree(d_target_ctx);
        cudaFree(d_ctxs);
        cudaFree(d_orig_sizes);
        cudaFree(d_pre_processed_sizes);
        cudaFree(d_culmn_sizes);
        cudaFree(d_pre_processed_msgs);
        delete[] h_pre_processed_msgs;
        return -1;
    }
    // Finalize: pack the contexts into digests
    md5_compare_ctx_batched<<<blocks_per_grid, threads_per_block>>>(d_ctxs, d_target_ctx, d_match_idx);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_match_idx);
        cudaFree(d_target_ctx);
        cudaFree(d_ctxs);
        cudaFree(d_orig_sizes);
        cudaFree(d_pre_processed_sizes);
        cudaFree(d_culmn_sizes);
        cudaFree(d_pre_processed_msgs);
        delete[] h_pre_processed_msgs;
        return -1;
    }
    // Copy into output match
    if (cudaMemcpy(&h_match_idx, d_match_idx, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        h_match_idx = -1;
    }
    // Free memory
    cudaFree(d_culmn_sizes);
    cudaFree(d_pre_processed_msgs);
    cudaFree(d_pre_processed_sizes);
    cudaFree(d_orig_sizes);
    cudaFree(d_ctxs);
    cudaFree(d_target_ctx);
    cudaFree(d_match_idx);
    delete[] h_pre_processed_msgs;

    return h_match_idx;
}

extern "C" {
    void init() {
        // Initialize the batched init_ctx
        for (int i = 0; i < BATCH_SIZE; i++) {
            init_ctxs[i].a = A0;
            init_ctxs[i].b = B0;
            init_ctxs[i].c = C0;
            init_ctxs[i].d = D0;
        }
    }

    int md5_target_batched_wrapper(FfiVectorBatch *msgs, FfiVector *target) {
        md5_ctx *target_ctx = new md5_ctx;
        uint8_t *data = target->data;
        
        // Fill target context registers with the target digest's ones
        target_ctx->a = data[0] +
                       (data[1] << 8) +
                       (data[2] << 16) +
                       (data[3] << 24);
        target_ctx->b = data[4] +
                       (data[5] << 8) +
                       (data[6] << 16) +
                       (data[7] << 24);
        target_ctx->c = data[8] +
                       (data[9] << 8) +
                       (data[10] << 16) +
                       (data[11] << 24);
        target_ctx->d = data[12] +
                       (data[13] << 8) +
                       (data[14] << 16) +
                       (data[15] << 24);
        
        int result = md5_target_batched(msgs->data, target_ctx);
        delete target_ctx;
        return result;
    }
}
