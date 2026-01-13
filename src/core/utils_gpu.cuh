#ifndef CUDA_UTILS
#define CUDA_UTILS

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#ifndef GPU_ERR_CHK
#define GPU_ERR_CHK(ans)                      \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
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
        exit(code);
    }
}
#endif

#endif
