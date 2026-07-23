#include <math.h>

#include "operators.h"

/* Element-wise CPU implementations, generated from operators.def.
 * Each macro expands one table entry into a full function body. */
#define EMBER_BINARY_OP(name, expr)                                          \
    void name##_tensor(const float *a, const float *b, float *out, int size) \
    {                                                                        \
        for (int i = 0; i < size; i++) out[i] = (expr);                      \
    }

#define EMBER_SCALAR_OP(name, expr)                                   \
    void name##_scalar(const float *a, float b, float *out, int size) \
    {                                                                 \
        for (int i = 0; i < size; i++) out[i] = (expr);               \
    }

#define EMBER_UNARY_OP(name, expr)                           \
    void name##_tensor(const float *a, float *out, int size) \
    {                                                        \
        for (int i = 0; i < size; i++) out[i] = (expr);      \
    }

#define EMBER_BROADCAST_OP(name, expr)                                                    \
    void name##_broadcasted(const float *a, const float *b, float *out, const int *shape, \
                            const int *strides_a, const int *strides_b, int ndim)         \
    {                                                                                     \
        int total = 1;                                                                    \
        for (int d = 0; d < ndim; d++) total *= shape[d];                                 \
        for (int i = 0; i < total; i++) {                                                 \
            int rem = i, ia = 0, ib = 0;                                                  \
            for (int d = ndim - 1; d >= 0; d--) {                                         \
                int coord = rem % shape[d];                                               \
                rem /= shape[d];                                                          \
                ia += coord * strides_a[d];                                               \
                ib += coord * strides_b[d];                                               \
            }                                                                             \
            out[i] = (expr);                                                              \
        }                                                                                 \
    }

#include "operators.def"

/* ---- non-element-wise operators ---- */
void matmul(const float *a, const float *b, float *out, int n, int m, int k)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            // Accumulate in double to keep the naive kernel's rounding error
            // close to a BLAS reference (the GPU backend uses cuBLAS).
            double acc = 0.0;
            for (int l = 0; l < k; l++) {
                acc += (double)a[i * k + l] * (double)b[l * m + j];
            }
            out[i * m + j] = (float)acc;
        }
    }
}

void transpose(const float *a, float *out, int n, int m)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            out[j * n + i] = a[i * m + j];
        }
    }
}

float sum(const float *a, int size)
{
    float s = 0.0f;
    for (int i = 0; i < size; i++) s += a[i];
    return s;
}

int sum_axis_product(const int *shape, int start, int end)
{
    int p = 1;
    for (int i = start; i < end; i++) p *= shape[i];
    return p;
}

void sum_axis(const float *a, float *out, int outer_stride, int inner_stride, int axis_dim)
{
    for (int o = 0; o < outer_stride; o++) {
        for (int i = 0; i < inner_stride; i++) {
            float s = 0.0f;
            int input_base = o * (axis_dim * inner_stride) + i;

            for (int r = 0; r < axis_dim; r++) {
                s += a[input_base + (r * inner_stride)];
            }
            out[o * inner_stride + i] = s;
        }
    }
}

void max_axis(const float *a, float *out, int outer_stride, int inner_stride, int axis_dim)
{
    for (int o = 0; o < outer_stride; o++) {
        for (int i = 0; i < inner_stride; i++) {
            int input_base = o * (axis_dim * inner_stride) + i;
            float m = a[input_base];

            for (int r = 1; r < axis_dim; r++) {
                m = fmaxf(m, a[input_base + (r * inner_stride)]);
            }
            out[o * inner_stride + i] = m;
        }
    }
}
