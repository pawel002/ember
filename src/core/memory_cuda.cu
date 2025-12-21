#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "memory.h"

#define CHECK_CUDA(call)                                                                    \
    {                                                                                       \
        cudaError_t err = call;                                                             \
        if (err != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, \
                    __LINE__);                                                              \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    }

extern "C" {

void *alloc_gpu(size_t bytes)
{
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, bytes);

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA OOM: Failed to allocate %zu bytes. Error: %s\n", bytes,
                cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
}

void free_gpu(void *ptr)
{
    if (ptr != NULL) {
        CHECK_CUDA(cudaFree(ptr));
    }
}

void copy_to_device(void *dst_device, const void *src_host, size_t bytes)
{
    CHECK_CUDA(cudaMemcpy(dst_device, src_host, bytes, cudaMemcpyHostToDevice));
}

void copy_from_device(void *dst_host, const void *src_device, size_t bytes)
{
    CHECK_CUDA(cudaMemcpy(dst_host, src_device, bytes, cudaMemcpyDeviceToHost));
}
}
