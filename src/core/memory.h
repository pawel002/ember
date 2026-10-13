#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h>  // for size_t

#ifdef __cplusplus
extern "C" {
#endif

void *alloc_memory(size_t bytes);
void free_memory(void *ptr);
void empty_device_cache(void);
/* Bytes of device memory currently held by live allocations (not counting
 * freed blocks kept in the cache). CPU backend: always 0 (not tracked). */
size_t device_memory_in_use(void);
void copy_to_device(void *dst_device, const void *src_host, size_t bytes);
void copy_from_device(void *dst_host, const void *src_device, size_t bytes);
/* Device->device copy, queued on the ember stream (capturable). */
void copy_device_to_device(void *dst_device, const void *src_device, size_t bytes);
/* Copy host->device via a reusable pinned staging buffer with an async copy on
 * the ember stream (faster H2D, and lets fresh batches be fed into a stable
 * device buffer between graph replays). CPU backend: a plain memcpy. */
void copy_to_device_pinned(void *dst_device, const void *src_host, size_t bytes);
void sync_device(void);
/* 1 if this is the CUDA backend, 0 for the CPU backend. */
int backend_is_cuda(void);

/* Matmul precision. 0 = full fp32 (cuBLAS default), 1 = TF32 tensor cores
 * (~3x the fp32 GEMM throughput on Ampere+, 10-bit mantissa on the inputs,
 * fp32 accumulate). Applies to every cuBLAS GEMM in the backend. CPU backend:
 * stores the flag but ignores it. */
void set_matmul_tf32(int enabled);
int get_matmul_tf32(void);

/* CUDA-graph capture of the ember stream. On the CPU backend these are no-ops
 * (end_capture returns NULL). end_capture returns an opaque cudaGraphExec_t. */
void begin_capture(void);
void *end_capture(void);
void graph_launch(void *exec);
void graph_destroy(void *exec);

#ifdef __cplusplus
}
#endif

#endif
