#include <math.h>

#include "operators.h"

#define BINARY_OP(NAME, EXPRESSION)                                          \
    void NAME##_tensor(const float *a, const float *b, float *out, int size) \
    {                                                                        \
        for (int i = 0; i < size; i++) out[i] = EXPRESSION;                  \
    }

#define SCALAR_OP(NAME, EXPRESSION)                                         \
    void NAME##_scalar(const float *a, const float b, float *out, int size) \
    {                                                                       \
        for (int i = 0; i < size; i++) out[i] = EXPRESSION;                 \
    }

#define INPLACE_OP(NAME, EXPRESSION)                        \
    void NAME##_inplace(float *a, const float *b, int size) \
    {                                                       \
        for (int i = 0; i < size; i++) a[i] = EXPRESSION;   \
    }

#define UNARY_OP(NAME, EXPRESSION)                           \
    void NAME##_tensor(const float *a, float *out, int size) \
    {                                                        \
        for (int i = 0; i < size; i++) out[i] = EXPRESSION;  \
    }

#define BROADCAST_OP(NAME, OPERATION)                                                     \
    void NAME##_broadcasted(const float *a, const float *b, float *out, const int *shape, \
                            const int *strides_a, const int *strides_b, int ndim)         \
    {                                                                                     \
        int total_elements = 1;                                                           \
        for (int i = 0; i < ndim; i++) {                                                  \
            total_elements *= shape[i];                                                   \
        }                                                                                 \
        for (int i = 0; i < total_elements; i++) {                                        \
            int temp_idx = i;                                                             \
            int offset_a = 0;                                                             \
            int offset_b = 0;                                                             \
            for (int d = ndim - 1; d >= 0; d--) {                                         \
                int coord = temp_idx % shape[d];                                          \
                temp_idx /= shape[d];                                                     \
                offset_a += coord * strides_a[d];                                         \
                offset_b += coord * strides_b[d];                                         \
            }                                                                             \
                                                                                          \
            out[i] = OPERATION;                                                           \
        }                                                                                 \
    }

// tensor operator implementations
BINARY_OP(add, a[i] + b[i])
BINARY_OP(sub, a[i] - b[i])
BINARY_OP(mul, a[i] * b[i])
BINARY_OP(truediv, a[i] / b[i])
BINARY_OP(max, fmaxf(a[i], b[i]))
BINARY_OP(min, fminf(a[i], b[i]))
BINARY_OP(gt, (a[i] > b[i]) ? 1.0f : 0.0f)
BINARY_OP(lt, (a[i] < b[i]) ? 1.0f : 0.0f)
BINARY_OP(pow, powf(a[i], b[i]))

// scalar operator implementations
SCALAR_OP(add, a[i] + b)
SCALAR_OP(sub, a[i] - b)
SCALAR_OP(rsub, b - a[i])
SCALAR_OP(mul, a[i] * b)
SCALAR_OP(rtruediv, b / a[i])
SCALAR_OP(max, fmaxf(a[i], b))
SCALAR_OP(min, fminf(a[i], b))
SCALAR_OP(gt, (a[i] > b) ? 1.0f : 0.0f)
SCALAR_OP(lt, (a[i] < b) ? 1.0f : 0.0f)
SCALAR_OP(pow, powf(a[i], b))
SCALAR_OP(rpow, powf(b, a[i]))

// optimization
void truediv_scalar(const float *a, const float b, float *out, int size)
{
    float inv_b = 1.0f / b;
    for (int i = 0; i < size; i++) out[i] = a[i] * inv_b;
}

// broadcasted operator implementations
BROADCAST_OP(add, a[offset_a] + b[offset_b])
BROADCAST_OP(sub, a[offset_a] - b[offset_b])
BROADCAST_OP(mul, a[offset_a] * b[offset_b])
BROADCAST_OP(truediv, a[offset_a] / b[offset_b])

// unary implementations
UNARY_OP(negate, -a[i])
UNARY_OP(exponent, expf(a[i]))
UNARY_OP(sqrt, sqrtf(a[i]))

UNARY_OP(sin, sinf(a[i]))
UNARY_OP(cos, cosf(a[i]))
UNARY_OP(tan, tanf(a[i]))
UNARY_OP(ctg, 1.0f / tanf(a[i]))

UNARY_OP(sinh, sinhf(a[i]))
UNARY_OP(cosh, coshf(a[i]))
UNARY_OP(tanh, tanhf(a[i]))
UNARY_OP(ctgh, 1.0f / tanhf(a[i]))

// inplace implementations
INPLACE_OP(isub, a[i] - b[i])

// cleanup macros
#undef BINARY_OP
#undef SCALAR_OP
#undef UNARY_OP
#undef BROADCAST_OP

// misc implementations
void matmul(const float *a, const float *b, float *out, int n, int m, int k)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * m + j];
            }
            out[i * m + j] = sum;
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

int sum_axis_product(int *shape, int start, int end)
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
