#include <math.h>

#include "ops.h"

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

void mul_scalar(const float *a, const float b, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = a[i] * b;
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

// misc operators
void negate(const float *a, float *out, int size)
{
    for (int i = 0; i < size; i++) out[i] = a[i] * -1.0f;
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
