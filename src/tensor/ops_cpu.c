#include <math.h>

#include "ops.h"

void add(const float *a, const float *b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = a[i] + b[i];
}

void subtract(const float *a, const float *b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = a[i] - b[i];
}

void multiply_elementwise(const float *a, const float *b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = a[i] * b[i];
}

void negate(const float *a, float *b, int size)
{
    for (int i = 0; i < size; i++) b[i] = a[i] * -1.0f;
}

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

void max_tensor(const float *a, const float *b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = fmaxf(a[i], b[i]);
}

void max_scalar(const float *a, const float b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = fmaxf(a[i], b);
}
