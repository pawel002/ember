#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h> // for size_t

#ifdef __cplusplus
extern "C" {
#endif

    void* alloc_gpu(size_t bytes);
    void free_gpu(void* ptr);
    void copy_to_gpu(void* dst_device, const void* src_host, size_t bytes);
    void copy_to_cpu(void* dst_host, const void* src_device, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif