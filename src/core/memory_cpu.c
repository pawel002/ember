#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "memory.h"

void *alloc_memory(size_t bytes)
{
    return malloc(bytes);
}

void free_memory(void *ptr)
{
    if (ptr != NULL) {
        free(ptr);
    }
}

void empty_device_cache(void)
{
    // No caching allocator on the CPU backend; malloc/free is used directly.
}

void copy_to_device(void *dst_device, const void *src_host, size_t bytes)
{
    memcpy(dst_device, src_host, bytes);
}

void copy_from_device(void *dst_host, const void *src_device, size_t bytes)
{
    memcpy(dst_host, src_device, bytes);
}

void copy_to_device_pinned(void *dst_device, const void *src_host, size_t bytes)
{
    memcpy(dst_device, src_host, bytes);
}

void sync_device() {}

// CUDA-graph capture is a no-op on the CPU backend.
void begin_capture(void) {}
void *end_capture(void)
{
    return NULL;
}
void graph_launch(void *exec)
{
    (void)exec;
}
void graph_destroy(void *exec)
{
    (void)exec;
}

/* The CPU backend has no cuBLAS; the flag is stored so the Python API behaves
 * the same on both backends. */
static int g_tf32 = 0;

void set_matmul_tf32(int enabled)
{
    g_tf32 = enabled ? 1 : 0;
}

int get_matmul_tf32(void)
{
    return g_tf32;
}
