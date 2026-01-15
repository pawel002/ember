#include <math.h>

#include "operators.h"

// tensor operators
void add_tensor(const float *a, const float *b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = a[i] + b[i];
}

void sub_tensor(const float *a, const float *b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = a[i] - b[i];
}

void mul_tensor(const float *a, const float *b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = a[i] * b[i];
}

void truediv_tensor(const float *a, const float *b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = a[i] / b[i];
}

void max_tensor(const float *a, const float *b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = fmaxf(a[i], b[i]);
}

void min_tensor(const float *a, const float *b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = fminf(a[i], b[i]);
}

void gt_tensor(const float *a, const float *b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
}

// scalar operators
void add_scalar(const float *a, const float b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = a[i] + b;
}

void sub_scalar(const float *a, const float b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = a[i] - b;
}

void rsub_scalar(const float *a, const float b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = b - a[i];
}

void mul_scalar(const float *a, const float b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = a[i] * b;
}

void truediv_scalar(const float *a, const float b, float *out, int size)
{
    float inv_b = 1.0f / b;
    for (int i = 0; i < size; i++) out[i] = a[i] * inv_b;
}

void rtruediv_scalar(const float *a, const float b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = b / a[i];
}

void max_scalar(const float *a, const float b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = fmaxf(a[i], b);
}

void min_scalar(const float *a, const float b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = fminf(a[i], b);
}

void gt_scalar(const float *a, const float b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = (a[i] > b) ? 1.0f : 0.0f;
}

// unary operators
void negate(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = a[i] * -1.0f;
}

void exponent(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = expf(a[i]);
}

// unary trigonometric
void sin_tensor(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = sinf(a[i]);
}

void cos_tensor(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = cosf(a[i]);
}

void tan_tensor(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = tanf(a[i]);
}

void ctg_tensor(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = 1.0f / tanf(a[i]);
}

// unary trigonometric hyperbolic
void sinh_tensor(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = sinhf(a[i]);
}

void cosh_tensor(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = coshf(a[i]);
}

void tanh_tensor(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = tanhf(a[i]);
}

void ctgh_tensor(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = 1.0f / tanhf(a[i]);
}

// misc operators
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
    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += a[i];
    return sum;
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
            float sum = 0.0f;
            int input_base = o * (axis_dim * inner_stride) + i;

            for (int r = 0; r < axis_dim; r++) {
                sum += a[input_base + (r * inner_stride)];
            }

            out[o * inner_stride + i] = sum;
        }
    }
}
