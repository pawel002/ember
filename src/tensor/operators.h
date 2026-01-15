#ifndef OPS_H
#define OPS_H

#ifdef __cplusplus
extern "C" {
#endif

// tensor operators
void add_tensor(const float *a, const float *b, float *out, int size);
void sub_tensor(const float *a, const float *b, float *out, int size);
void mul_tensor(const float *a, const float *b, float *out, int size);
void truediv_tensor(const float *a, const float *b, float *out, int size);
void max_tensor(const float *a, const float *b, float *out, int size);
void min_tensor(const float *a, const float *b, float *out, int size);
void gt_tensor(const float *a, const float *b, float *out, int size);

// scalar oprator
void add_scalar(const float *a, const float b, float *out, int size);
void sub_scalar(const float *a, const float b, float *out, int size);
void rsub_scalar(const float *a, const float b, float *out, int size);
void mul_scalar(const float *a, const float b, float *out, int size);
void truediv_scalar(const float *a, const float b, float *out, int size);
void rtruediv_scalar(const float *a, const float b, float *out, int size);
void max_scalar(const float *a, const float b, float *out, int size);
void min_scalar(const float *a, const float b, float *out, int size);
void gt_scalar(const float *a, const float b, float *out, int size);

// unary operators
void negate(const float *a, float *out, int size);
void exponent(const float *a, float *out, int size);

// unary trigonometric
void sin_tensor(const float *a, float *out, int size);
void cos_tensor(const float *a, float *out, int size);
void tan_tensor(const float *a, float *out, int size);
void ctg_tensor(const float *a, float *out, int size);

// unary trigonometric hyperbolic
void sinh_tensor(const float *a, float *out, int size);
void cosh_tensor(const float *a, float *out, int size);
void tanh_tensor(const float *a, float *out, int size);
void ctgh_tensor(const float *a, float *out, int size);

// misc operators
void matmul(const float *a, const float *b, float *out, int n, int m, int k);
void transpose(const float *a, float *out, int n, int m);
float sum(const float *a, int size);
int sum_axis_product(int *shape, int start, int end);
void sum_axis(const float *a, float *out, int outer_stride, int inner_stride, int axis_dim);

#ifdef __cplusplus
}
#endif

#endif
