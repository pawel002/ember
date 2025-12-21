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

void copy_to_device(void *dst_device, const void *src_host, size_t bytes)
{
    memcpy(dst_device, src_host, bytes);
}

void copy_from_device(void *dst_host, const void *src_device, size_t bytes)
{
    memcpy(dst_host, src_device, bytes);
}
