#ifndef OPS_H
#define OPS_H

#ifdef __cplusplus
extern "C" {
#endif

void add(const float *a, const float *b, float *out, int size);
void subtract(const float *a, const float *b, float *out, int size);
void multiply_elementwise(const float *a, const float *b, float *out, int size);
void simple_matmul(const float *a, const float *b, float *out, int n, int m, int k);
void negate(const float *a, float *b, int size);

#ifdef __cplusplus
}
#endif

#endif
