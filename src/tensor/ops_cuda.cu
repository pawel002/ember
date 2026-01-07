#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>

#include "../core/utils_cuda.cuh"
#include "ops.h"

#define BLOCK_SIZE 256

constexpr const float matmul_alpha = 1.0f;
constexpr const float matmul_beta = 0.0f;

// tensor vs tensor kernels
__global__ void k_add_tensor(const float *a, const float *b, float *out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = a[i] + b[i];
}

__global__ void k_sub_tensor(const float *a, const float *b, float *out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = a[i] - b[i];
}

__global__ void k_mul_tensor(const float *a, const float *b, float *out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = a[i] * b[i];
}

__global__ void k_max_tensor(const float *a, const float *b, float *out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = fmaxf(a[i], b[i]);
}

__global__ void k_min_tensor(const float *a, const float *b, float *out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = fminf(a[i], b[i]);
}

__global__ void k_gt_tensor(const float *a, const float *b, float *out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
}

// tensor vs scalar kernels
__global__ void k_add_scalar(const float *a, const float b, float *out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = a[i] + b;
}

__global__ void k_sub_scalar(const float *a, const float b, float *out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = a[i] - b;
}

__global__ void k_mul_scalar(const float *a, const float b, float *out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = a[i] * b;
}

__global__ void k_max_scalar(const float *a, const float b, float *out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = fmaxf(a[i], b);
}

__global__ void k_min_scalar(const float *a, const float b, float *out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = fminf(a[i], b);
}

__global__ void k_gt_scalar(const float *a, const float b, float *out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = (a[i] > b) ? 1.0f : 0.0f;
}

// misc kernels
__global__ void k_negate(const float *a, float *out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = -a[i];
}

// matmul
__global__ void k_simple_matmul(const float *a, const float *b, float *out, int n, int m, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < k) {
        float val = 0.0f;
        for (int p = 0; p < m; ++p) {
            val += a[row * m + p] * b[p * k + col];
        }
        out[row * k + col] = val;
    }
}

// grid size calc
static int get_grid_size(int n)
{
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

extern "C" {

// tensor wrappers
void add_tensor(const float *a, const float *b, float *out, int size)
{
    k_add_tensor<<<get_grid_size(size), BLOCK_SIZE>>>(a, b, out, size);
}

void sub_tensor(const float *a, const float *b, float *out, int size)
{
    k_sub_tensor<<<get_grid_size(size), BLOCK_SIZE>>>(a, b, out, size);
}

void mul_tensor(const float *a, const float *b, float *out, int size)
{
    k_mul_tensor<<<get_grid_size(size), BLOCK_SIZE>>>(a, b, out, size);
}

void max_tensor(const float *a, const float *b, float *out, int size)
{
    k_max_tensor<<<get_grid_size(size), BLOCK_SIZE>>>(a, b, out, size);
}

void min_tensor(const float *a, const float *b, float *out, int size)
{
    k_min_tensor<<<get_grid_size(size), BLOCK_SIZE>>>(a, b, out, size);
}

void gt_tensor(const float *a, const float *b, float *out, int size)
{
    k_gt_tensor<<<get_grid_size(size), BLOCK_SIZE>>>(a, b, out, size);
}

// scalar wrappers
void add_scalar(const float *a, const float b, float *out, int size)
{
    k_add_scalar<<<get_grid_size(size), BLOCK_SIZE>>>(a, b, out, size);
}

void sub_scalar(const float *a, const float b, float *out, int size)
{
    k_sub_scalar<<<get_grid_size(size), BLOCK_SIZE>>>(a, b, out, size);
}

void mul_scalar(const float *a, const float b, float *out, int size)
{
    k_mul_scalar<<<get_grid_size(size), BLOCK_SIZE>>>(a, b, out, size);
}

void max_scalar(const float *a, const float b, float *out, int size)
{
    k_max_scalar<<<get_grid_size(size), BLOCK_SIZE>>>(a, b, out, size);
}

void min_scalar(const float *a, const float b, float *out, int size)
{
    k_min_scalar<<<get_grid_size(size), BLOCK_SIZE>>>(a, b, out, size);
}

void gt_scalar(const float *a, const float b, float *out, int size)
{
    k_gt_scalar<<<get_grid_size(size), BLOCK_SIZE>>>(a, b, out, size);
}

// misc wrappers
void negate(const float *a, float *out, int size)
{
    k_negate<<<get_grid_size(size), BLOCK_SIZE>>>(a, out, size);
}

void simple_matmul(const float *a, const float *b, float *out, int n, int m, int k)
{
    // TODO: move handle to global application context
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, b, m, a, k, &beta, out, m);

    cublasDestroy(handle);
}
}
