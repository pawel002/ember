#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "memory.h"
#include "utils_cuda.cuh"

extern "C" {

void *alloc_memory(size_t bytes)
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

void free_memory(void *ptr)
{
    if (ptr != NULL) {
        GPU_ERR_CHK(cudaFree(ptr));
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
