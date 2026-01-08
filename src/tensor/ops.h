#ifndef OPS_H
#define OPS_H

#ifdef __cplusplus
extern "C" {
#endif

// tensor operators
void add_tensor(const float *a, const float *b, float *out, int size);
void sub_tensor(const float *a, const float *b, float *out, int size);
void mul_tensor(const float *a, const float *b, float *out, int size);
void max_tensor(const float *a, const float *b, float *out, int size);
void min_tensor(const float *a, const float *b, float *out, int size);
void gt_tensor(const float *a, const float *b, float *out, int size);

// scalar oprator
void add_scalar(const float *a, const float b, float *out, int size);
void sub_scalar(const float *a, const float b, float *out, int size);
void rsub_scalar(const float *a, const float b, float *out, int size);
void mul_scalar(const float *a, const float b, float *out, int size);
void max_scalar(const float *a, const float b, float *out, int size);
void min_scalar(const float *a, const float b, float *out, int size);
void gt_scalar(const float *a, const float b, float *out, int size);

// misc operators
void simple_matmul(const float *a, const float *b, float *out, int n, int m, int k);
void negate(const float *a, float *out, int size);

#ifdef __cplusplus
}
#endif

#endif
