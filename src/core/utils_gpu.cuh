#ifndef CUDA_UTILS
#define CUDA_UTILS

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef GPU_ERR_CHK
#define GPU_ERR_CHK(ans)                      \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(EXIT_FAILURE);
    }
}
#endif

#ifndef CUBLAS_ERR_CHK
#define CUBLAS_ERR_CHK(ans)                      \
    {                                            \
        cublasAssert((ans), __FILE__, __LINE__); \
    }
inline void cublasAssert(cublasStatus_t code, const char *file, int line)
{
    if (code != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLASassert: Error Code %d %s %d\n", code, file, line);
        exit(EXIT_FAILURE);
    }
}
#endif

// Check for launch-configuration errors immediately after a kernel launch.
// Asynchronous execution errors surface at the next device synchronization
// (which happens on every host<->device transfer).
#define CUDA_POST_LAUNCH() GPU_ERR_CHK(cudaGetLastError())

// Threads per block for the 1-D element-wise kernels and the block-per-row
// kernels. Must stay a power of two (block_reduce_* below rely on it).
#define EMBER_BLOCK 256

// Blocks needed to cover n elements with one thread each.
static inline int ember_grid(int n)
{
    return (n + EMBER_BLOCK - 1) / EMBER_BLOCK;
}

// The single CUDA stream all ember GPU work runs on. Using a dedicated
// (non-default) stream is what makes the training step capturable into a CUDA
// graph. Defined in memory_gpu.cu.
cudaStream_t ember_stream(void);

// The process-wide cuBLAS handle, already bound to the ember stream and to the
// selected matmul precision (see set_matmul_tf32 in memory.h). Fetch it right
// before each cuBLAS call so a precision toggle takes effect immediately.
// Defined in memory_gpu.cu.
cublasHandle_t ember_cublas(void);

#ifdef __CUDACC__
// Sum / max of `v` over the block, returned to every thread. `sh` is shared
// scratch of blockDim.x floats; blockDim.x must be a power of two.
static __device__ __forceinline__ float block_reduce_sum(float v, float *sh)
{
    int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    float r = sh[0];
    __syncthreads();
    return r;
}

static __device__ __forceinline__ float block_reduce_max(float v, float *sh)
{
    int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    float r = sh[0];
    __syncthreads();
    return r;
}

// Sum of `v` over the warp, left in lane 0.
static __device__ __forceinline__ float warp_reduce_sum(float v)
{
    for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}
#endif

#endif
