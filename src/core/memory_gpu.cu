#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <unordered_map>
#include <vector>

#include "memory.h"
#include "utils_gpu.cuh"

// Caching device allocator.
//
// A fixed-shape training loop allocates and frees the same set of buffer sizes
// every iteration. Calling cudaMalloc/cudaFree each time is very expensive
// (both synchronize with the device). Instead we recycle freed blocks by exact
// byte size: after warmup the loop stops touching the driver allocator
// entirely, and -- because the alloc/free pattern is deterministic -- it hands
// back the *same* addresses each iteration, which is what CUDA-graph capture
// requires.
//
// All calls happen under the Python GIL (single-threaded), so no locking is
// needed. Freed memory is retained for the process lifetime (a pool that only
// grows to the loop's steady-state footprint); empty_device_cache() can release
// it back to the driver.

namespace
{
std::unordered_map<size_t, std::vector<void *>> g_free_lists;
std::unordered_map<void *, size_t> g_block_size;
}  // namespace

extern "C" {

void *alloc_memory(size_t bytes)
{
    if (bytes == 0) return NULL;

    auto it = g_free_lists.find(bytes);
    if (it != g_free_lists.end() && !it->second.empty()) {
        void *ptr = it->second.back();
        it->second.pop_back();
        return ptr;
    }

    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA OOM: Failed to allocate %zu bytes. Error: %s\n", bytes,
                cudaGetErrorString(err));
        return NULL;
    }

    g_block_size[ptr] = bytes;
    return ptr;
}

void free_memory(void *ptr)
{
    if (ptr == NULL) return;

    auto it = g_block_size.find(ptr);
    if (it == g_block_size.end()) {
        // Not from this allocator (should not happen) -- release directly.
        GPU_ERR_CHK(cudaFree(ptr));
        return;
    }

    // Return the block to its size bucket for reuse (kept in g_block_size).
    g_free_lists[it->second].push_back(ptr);
}

void empty_device_cache(void)
{
    for (auto &kv : g_free_lists) {
        for (void *ptr : kv.second) {
            g_block_size.erase(ptr);
            GPU_ERR_CHK(cudaFree(ptr));
        }
        kv.second.clear();
    }
}

void copy_to_device(void *dst_device, const void *src_host, size_t bytes)
{
    GPU_ERR_CHK(cudaMemcpy(dst_device, src_host, bytes, cudaMemcpyHostToDevice));
}

void copy_from_device(void *dst_host, const void *src_device, size_t bytes)
{
    GPU_ERR_CHK(cudaMemcpy(dst_host, src_device, bytes, cudaMemcpyDeviceToHost));
}

void sync_device()
{
    GPU_ERR_CHK(cudaDeviceSynchronize());
}
}
