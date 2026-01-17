#ifndef OPS_H
#define OPS_H

#ifdef __cplusplus
extern "C" {
#endif

#define BINARY_OP(NAME) void NAME##_tensor(const float *a, const float *b, float *out, int size)
#define SCALAR_OP(NAME) void NAME##_scalar(const float *a, const float b, float *out, int size)
#define UNARY_OP(NAME) void NAME##_tensor(const float *a, float *out, int size)
#define BROADCAST_OP(NAME)                                                                \
    void NAME##_broadcasted(const float *a, const float *b, float *out, const int *shape, \
                            const int *strides_a, const int *strides_b, int dim);

// tensor operations
BINARY_OP(add);
BINARY_OP(sub);
BINARY_OP(mul);
BINARY_OP(truediv);
BINARY_OP(max);
BINARY_OP(min);
BINARY_OP(gt);
BINARY_OP(lt);
BINARY_OP(pow);

// scalar operations
SCALAR_OP(add);
SCALAR_OP(sub);
SCALAR_OP(rsub);
SCALAR_OP(mul);
SCALAR_OP(truediv);
SCALAR_OP(rtruediv);
SCALAR_OP(max);
SCALAR_OP(min);
SCALAR_OP(gt);
SCALAR_OP(lt);
SCALAR_OP(pow);
SCALAR_OP(rpow);

// broadcast operations
BROADCAST_OP(add);
BROADCAST_OP(sub);
BROADCAST_OP(mul);
BROADCAST_OP(truediv);

// unary operations
UNARY_OP(negate);
UNARY_OP(exponent);
UNARY_OP(sqrt);

UNARY_OP(sin);
UNARY_OP(cos);
UNARY_OP(tan);
UNARY_OP(ctg);

UNARY_OP(sinh);
UNARY_OP(cosh);
UNARY_OP(tanh);
UNARY_OP(ctgh);

// misc operations
void matmul(const float *a, const float *b, float *out, int n, int m, int k);
void transpose(const float *a, float *out, int n, int m);
float sum(const float *a, int size);
int sum_axis_product(int *shape, int start, int end);
void sum_axis(const float *a, float *out, int outer_stride, int inner_stride, int axis_dim);

#undef BINARY_OP
#undef SCALAR_OP
#undef UNARY_OP
#undef BROADCAST_OP

#ifdef __cplusplus
}
#endif

#endif
