#include <math.h>
#include <stddef.h>  // size_t

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

#define EMBER_INPLACE_OP(name, expr)                        \
    void name##_inplace(float *a, const float *b, int size) \
    {                                                       \
        for (int i = 0; i < size; i++) a[i] = (expr);       \
    }

#define EMBER_INPLACE_SCALAR_OP(name, expr)                 \
    void name##_scalar_inplace(float *a, float b, int size) \
    {                                                       \
        for (int i = 0; i < size; i++) a[i] = (expr);       \
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

void matmul_batched(const float *a, const float *b, float *out, int batch, int n, int m, int k)
{
    for (int bi = 0; bi < batch; bi++) {
        matmul(a + (size_t)bi * n * k, b + (size_t)bi * k * m, out + (size_t)bi * n * m, n, m, k);
    }
}

void matmul_bias(const float *a, const float *b, const float *bias, float *out, int n, int m, int k)
{
    matmul(a, b, out, n, m, k);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            out[i * m + j] += bias[j];
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

void gelu_bwd(const float *grad, const float *x, float *out, int n)
{
    const float a = 0.8f;
    for (int i = 0; i < n; i++) {
        float t = tanhf(a * x[i]);
        out[i] = grad[i] * 0.5f * (1.0f + t) * (1.0f + a * x[i] * (1.0f - t));
    }
}

/* ---- fused optimizer steps ----
 * mb1/mb2 are 1-beta1/1-beta2 and bc1/bc2 are the bias corrections
 * 1/(1-beta^t), both precomputed on the host (in double, then cast) so the
 * kernel reproduces the op-by-op float32 result exactly. */
void adam_step(float *p, const float *g, float *m, float *v, int size, float lr, float beta1,
               float mb1, float beta2, float mb2, float eps, float bc1, float bc2)
{
    for (int i = 0; i < size; i++) {
        float gi = g[i];
        float mi = beta1 * m[i] + mb1 * gi;
        float vi = beta2 * v[i] + mb2 * gi * gi;
        m[i] = mi;
        v[i] = vi;
        p[i] -= lr * (mi * bc1) / (sqrtf(vi * bc2) + eps);
    }
}

void adamw_step(float *p, const float *g, float *m, float *v, int size, float lr, float beta1,
                float mb1, float beta2, float mb2, float eps, float bc1, float bc2,
                float weight_decay)
{
    for (int i = 0; i < size; i++) {
        float gi = g[i];
        float mi = beta1 * m[i] + mb1 * gi;
        float vi = beta2 * v[i] + mb2 * gi * gi;
        m[i] = mi;
        v[i] = vi;
        p[i] = p[i] * (1.0f - lr * weight_decay) - lr * (mi * bc1) / (sqrtf(vi * bc2) + eps);
    }
}

void sgd_step(float *p, const float *g, float *v, int size, float lr, float momentum)
{
    for (int i = 0; i < size; i++) {
        float vi = momentum * v[i] + g[i];
        v[i] = vi;
        p[i] -= lr * vi;
    }
}

void adam_step_group(float **ps, float **gs, float **ms, float **vs, const int *sizes, int nparams,
                     int max_size, float lr, float beta1, float mb1, float beta2, float mb2,
                     float eps, float bc1, float bc2)
{
    (void)max_size;
    for (int pi = 0; pi < nparams; pi++) {
        adam_step(ps[pi], gs[pi], ms[pi], vs[pi], sizes[pi], lr, beta1, mb1, beta2, mb2, eps, bc1,
                  bc2);
    }
}

void adamw_step_group(float **ps, float **gs, float **ms, float **vs, const int *sizes, int nparams,
                      int max_size, float lr, float beta1, float mb1, float beta2, float mb2,
                      float eps, float bc1, float bc2, float weight_decay)
{
    (void)max_size;
    for (int pi = 0; pi < nparams; pi++) {
        adamw_step(ps[pi], gs[pi], ms[pi], vs[pi], sizes[pi], lr, beta1, mb1, beta2, mb2, eps, bc1,
                   bc2, weight_decay);
    }
}

void adam_bias_update(float *t, float *bc, float beta1, float beta2)
{
    float tt = t[0] + 1.0f;
    t[0] = tt;
    bc[0] = 1.0f / (1.0f - powf(beta1, tt));
    bc[1] = 1.0f / (1.0f - powf(beta2, tt));
}

void adam_step_dev(float *p, const float *g, float *m, float *v, int size, float lr, float beta1,
                   float mb1, float beta2, float mb2, float eps, const float *bc)
{
    for (int i = 0; i < size; i++) {
        float gi = g[i];
        float mi = beta1 * m[i] + mb1 * gi;
        float vi = beta2 * v[i] + mb2 * gi * gi;
        m[i] = mi;
        v[i] = vi;
        p[i] -= lr * (mi * bc[0]) / (sqrtf(vi * bc[1]) + eps);
    }
}

void adamw_step_dev(float *p, const float *g, float *m, float *v, int size, float lr, float beta1,
                    float mb1, float beta2, float mb2, float eps, const float *bc,
                    float weight_decay)
{
    for (int i = 0; i < size; i++) {
        float gi = g[i];
        float mi = beta1 * m[i] + mb1 * gi;
        float vi = beta2 * v[i] + mb2 * gi * gi;
        m[i] = mi;
        v[i] = vi;
        p[i] = p[i] * (1.0f - lr * weight_decay) - lr * (mi * bc[0]) / (sqrtf(vi * bc[1]) + eps);
    }
}
