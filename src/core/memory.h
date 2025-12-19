#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h> // for size_t

#ifdef __cplusplus
extern "C" {
#endif

    void* alloc_memory(size_t bytes);
    void free_memory(void* ptr);
    void copy_to_device(void* dst_device, const void* src_host, size_t bytes);
    void copy_from_device(void* dst_host, const void* src_device, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif