#include "ops.h"

void add(const float* a, const float* b, float* out, int size) {
    for(int i = 0; i < size; i++)
        out[i] = a[i] + b[i];
}

void subtract(const float* a, const float* b, float* out, int size) {
    for(int i = 0; i < size; i++)
        out[i] = a[i] - b[i];
}

void multiply_elementwise(const float* a, const float* b, float* out, int size) {
    for(int i = 0; i < size; i++)
        out[i] = a[i] * b[i];
}

void negate(float* a, int size) {
    for(int i = 0; i < size; i++)
        a[i] *= -1;
}

