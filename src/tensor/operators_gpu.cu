#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include "../core/memory.h"
#include "../core/utils_gpu.cuh"
#include "operators.h"

#define BLOCK_SIZE 256

static int grid(int n)
{
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

/* Element-wise CUDA kernels + host wrappers, generated from operators.def.
 * Kernels keep C++ linkage (called only within this file); the wrappers that
 * the rest of the library links against get C linkage. */
#define EMBER_BINARY_OP(name, expr)                                                    \
    __global__ void k_##name##_tensor(const float *a, const float *b, float *out,      \
                                      int size)                                        \
    {                                                                                  \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
        if (i < size) out[i] = (expr);                                                 \
    }                                                                                  \
    extern "C" void name##_tensor(const float *a, const float *b, float *out, int size) \
    {                                                                                  \
        k_##name##_tensor<<<grid(size), BLOCK_SIZE>>>(a, b, out, size);                \
    }

#define EMBER_SCALAR_OP(name, expr)                                                   \
    __global__ void k_##name##_scalar(const float *a, float b, float *out, int size)  \
    {                                                                                 \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                                \
        if (i < size) out[i] = (expr);                                                \
    }                                                                                 \
    extern "C" void name##_scalar(const float *a, float b, float *out, int size)      \
    {                                                                                 \
        k_##name##_scalar<<<grid(size), BLOCK_SIZE>>>(a, b, out, size);               \
    }

#define EMBER_UNARY_OP(name, expr)                                                    \
    __global__ void k_##name##_tensor(const float *a, float *out, int size)           \
    {                                                                                 \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                                \
        if (i < size) out[i] = (expr);                                                \
    }                                                                                 \
    extern "C" void name##_tensor(const float *a, float *out, int size)               \
    {                                                                                 \
        k_##name##_tensor<<<grid(size), BLOCK_SIZE>>>(a, out, size);                  \
    }

#define EMBER_BROADCAST_OP(name, expr)                                                \
    __global__ void k_##name##_broadcasted(const float *a, const float *b, float *out, \
                                           const int *shape, const int *strides_a,     \
                                           const int *strides_b, int ndim, int total)  \
    {                                                                                  \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                                 \
        if (i >= total) return;                                                        \
        int rem = i, ia = 0, ib = 0;                                                   \
        for (int d = ndim - 1; d >= 0; d--) {                                          \
            int coord = rem % shape[d];                                                \
            rem /= shape[d];                                                           \
            ia += coord * strides_a[d];                                                \
            ib += coord * strides_b[d];                                                \
        }                                                                              \
        out[i] = (expr);                                                               \
    }                                                                                  \
    extern "C" void name##_broadcasted(const float *a, const float *b, float *out,     \
                                       const int *shape, const int *strides_a,          \
                                       const int *strides_b, int ndim)                  \
    {                                                                                  \
        int total = 1;                                                                 \
        for (int d = 0; d < ndim; d++) total *= shape[d];                              \
        int *d_shape = (int *)alloc_memory(ndim * sizeof(int));                        \
        int *d_sa = (int *)alloc_memory(ndim * sizeof(int));                           \
        int *d_sb = (int *)alloc_memory(ndim * sizeof(int));                           \
        copy_to_device(d_shape, shape, ndim * sizeof(int));                            \
        copy_to_device(d_sa, strides_a, ndim * sizeof(int));                           \
        copy_to_device(d_sb, strides_b, ndim * sizeof(int));                           \
        k_##name##_broadcasted<<<grid(total), BLOCK_SIZE>>>(a, b, out, d_shape, d_sa,  \
                                                            d_sb, ndim, total);        \
        sync_device();                                                                 \
        free_memory(d_shape);                                                          \
        free_memory(d_sa);                                                             \
        free_memory(d_sb);                                                             \
    }

#include "operators.def"

/* ---- non-element-wise operators ---- */
extern "C" {

void matmul(const float *a, const float *b, float *out, int n, int m, int k)
{
    // C(n x m) = A(n x k) * B(k x m), row-major. cuBLAS is column-major, so we
    // compute C^T = B^T * A^T by swapping the operands and using the fact that
    // a row-major (n x m) buffer is a column-major (m x n) buffer.
    cublasHandle_t handle;
    CUBLAS_ERR_CHK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_ERR_CHK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, b, m, a, k,
                               &beta, out, m));

    CUBLAS_ERR_CHK(cublasDestroy(handle));
}

__global__ void k_transpose(const float *a, float *out, int n, int m)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < m) out[j * n + i] = a[i * m + j];
}

void transpose(const float *a, float *out, int n, int m)
{
    dim3 block(16, 16);
    dim3 g((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    k_transpose<<<g, block>>>(a, out, n, m);
}

float sum(const float *a, int size)
{
    // Simple, obviously-correct reduction via a host round-trip.
    float *host = (float *)malloc((size_t)size * sizeof(float));
    copy_from_device(host, a, (size_t)size * sizeof(float));

    float s = 0.0f;
    for (int i = 0; i < size; i++) s += host[i];

    free(host);
    return s;
}

int sum_axis_product(const int *shape, int start, int end)
{
    int p = 1;
    for (int i = start; i < end; i++) p *= shape[i];
    return p;
}

__global__ void k_sum_axis(const float *a, float *out, int outer_stride, int inner_stride,
                           int axis_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer_stride * inner_stride;
    if (idx >= total) return;

    int o = idx / inner_stride;
    int i = idx % inner_stride;
    int input_base = o * (axis_dim * inner_stride) + i;

    float s = 0.0f;
    for (int r = 0; r < axis_dim; r++) s += a[input_base + (r * inner_stride)];
    out[idx] = s;
}

void sum_axis(const float *a, float *out, int outer_stride, int inner_stride, int axis_dim)
{
    int total = outer_stride * inner_stride;
    k_sum_axis<<<grid(total), BLOCK_SIZE>>>(a, out, outer_stride, inner_stride, axis_dim);
}
}
