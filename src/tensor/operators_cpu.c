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
void t_sin(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = sinf(a[i]);
}

void t_cos(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = cosf(a[i]);
}

void t_tan(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = tanf(a[i]);
}

void t_ctg(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = 1.0f / tanf(a[i]);
}

// unary trigonometric hyperbolic
void t_sinh(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = sinhf(a[i]);
}

void t_cosh(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = coshf(a[i]);
}

void t_tanh(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = tanhf(a[i]);
}

void t_ctgh(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = 1.0f / tanhf(a[i]);
}

// misc operators
void simple_matmul(const float *a, const float *b, float *out, int n, int m, int k)
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
