#ifndef OPS_H
#define OPS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Element-wise operator declarations, generated from operators.def.
 * See operators.def for the operator table itself. */
#define EMBER_BINARY_OP(name, expr) \
    void name##_tensor(const float *a, const float *b, float *out, int size);
#define EMBER_SCALAR_OP(name, expr) \
    void name##_scalar(const float *a, float b, float *out, int size);
#define EMBER_BROADCAST_OP(name, expr)                                                    \
    void name##_broadcasted(const float *a, const float *b, float *out, const int *shape, \
                            const int *strides_a, const int *strides_b, int ndim);
#define EMBER_UNARY_OP(name, expr) void name##_tensor(const float *a, float *out, int size);
#define EMBER_INPLACE_OP(name, expr) void name##_inplace(float *a, const float *b, int size);
#define EMBER_INPLACE_SCALAR_OP(name, expr) void name##_scalar_inplace(float *a, float b, int size);
#include "operators.def"

/* Non-element-wise operators (hand-written per backend). */
void matmul(const float *a, const float *b, float *out, int n, int m, int k);
void matmul_batched(const float *a, const float *b, float *out, int batch, int n, int m, int k);
/* out = a @ b + bias, with bias broadcast over rows (length m). Fuses the
 * Linear epilogue: one GEMM + one cheap row-broadcast add, no broadcast temp. */
void matmul_bias(const float *a, const float *b, const float *bias, float *out, int n, int m,
                 int k);
void transpose(const float *a, float *out, int n, int m);
float sum(const float *a, int size);
int sum_axis_product(const int *shape, int start, int end);
void sum_axis(const float *a, float *out, int outer_stride, int inner_stride, int axis_dim);
void max_axis(const float *a, float *out, int outer_stride, int inner_stride, int axis_dim);

/* GELU backward needs (grad, x, y); the other activation backprops are binary
 * ops generated from operators.def (relu_bwd/sigmoid_bwd/tanh_bwd). */
void gelu_bwd(const float *grad, const float *x, const float *y, float *out, int n);

/* Fused, in-place optimizer steps (one kernel launch per parameter). The bias
 * corrections (bc1 = 1/(1-beta1^t), bc2 = 1/(1-beta2^t)) are precomputed on the
 * host so the kernel needs no per-step scalar powers. */
void adam_step(float *p, const float *g, float *m, float *v, int size, float lr, float beta1,
               float mb1, float beta2, float mb2, float eps, float bc1, float bc2);
void adamw_step(float *p, const float *g, float *m, float *v, int size, float lr, float beta1,
                float mb1, float beta2, float mb2, float eps, float bc1, float bc2,
                float weight_decay);
void sgd_step(float *p, const float *g, float *v, int size, float lr, float momentum);

/* Capturable Adam/AdamW: the step counter lives on-device (t, a length-1
 * buffer) and the bias corrections (bc, length-2: [bc1, bc2]) are computed
 * on-device, so a captured graph advances t on every replay and stays exact.
 * Call adam_bias_update once per step, then *_step_dev per parameter. */
void adam_bias_update(float *t, float *bc, float beta1, float beta2);
void adam_step_dev(float *p, const float *g, float *m, float *v, int size, float lr, float beta1,
                   float mb1, float beta2, float mb2, float eps, const float *bc);
void adamw_step_dev(float *p, const float *g, float *m, float *v, int size, float lr, float beta1,
                    float mb1, float beta2, float mb2, float eps, const float *bc,
                    float weight_decay);

/* Grouped (multi-tensor) Adam: one launch updates all `nparams` parameters.
 * ps/gs/ms/vs are arrays of per-parameter buffers and `sizes` their lengths
 * (all device-resident on the GPU backend). max_size is the largest length. */
void adam_step_group(float **ps, float **gs, float **ms, float **vs, const int *sizes, int nparams,
                     int max_size, float lr, float beta1, float mb1, float beta2, float mb2,
                     float eps, float bc1, float bc2);

#ifdef __cplusplus
}
#endif

#endif
