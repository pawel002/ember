#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h>  // for size_t

#ifdef __cplusplus
extern "C" {
#endif

void *alloc_memory(size_t bytes);
void free_memory(void *ptr);
void empty_device_cache(void);
void copy_to_device(void *dst_device, const void *src_host, size_t bytes);
void copy_from_device(void *dst_host, const void *src_device, size_t bytes);
/* Copy host->device via a reusable pinned staging buffer with an async copy on
 * the ember stream (faster H2D, and lets fresh batches be fed into a stable
 * device buffer between graph replays). CPU backend: a plain memcpy. */
void copy_to_device_pinned(void *dst_device, const void *src_host, size_t bytes);
void sync_device();

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
