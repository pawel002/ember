#include "ops.h"

void add(const float* a, const float* b, float* out, int size) {
    for(int i = 0; i < size; i++)
        out[i] = a[i] + b[i];
}