#define BINARY_OP(NAME, EXPRESSION)                                          \
    void NAME##_tensor(const float *a, const float *b, float *out, int size) \
    {                                                                        \
        for (int i = 0; i < size; i++) out[i] = EXPRESSION;                  \
    }

#define SCALAR_OP(NAME, EXPRESSION)                                         \
    void NAME##_scalar(const float *a, const float b, float *out, int size) \
    {                                                                       \
        for (int i = 0; i < size; i++) out[i] = EXPRESSION;                 \
    }

#define INPLACE_OP(NAME, EXPRESSION)                        \
    void NAME##_inplace(float *a, const float *b, int size) \
    {                                                       \
        for (int i = 0; i < size; i++) a[i] = EXPRESSION;   \
    }

#define UNARY_OP(NAME, EXPRESSION)                           \
    void NAME##_tensor(const float *a, float *out, int size) \
    {                                                        \
        for (int i = 0; i < size; i++) out[i] = EXPRESSION;  \
    }

#define BROADCAST_OP(NAME, OPERATION)                                                     \
    void NAME##_broadcasted(const float *a, const float *b, float *out, const int *shape, \
                            const int *strides_a, const int *strides_b, int ndim)         \
    {                                                                                     \
        int total_elements = 1;                                                           \
        for (int i = 0; i < ndim; i++) {                                                  \
            total_elements *= shape[i];                                                   \
        }                                                                                 \
        for (int i = 0; i < total_elements; i++) {                                        \
            int temp_idx = i;                                                             \
            int offset_a = 0;                                                             \
            int offset_b = 0;                                                             \
            for (int d = ndim - 1; d >= 0; d--) {                                         \
                int coord = temp_idx % shape[d];                                          \
                temp_idx /= shape[d];                                                     \
                offset_a += coord * strides_a[d];                                         \
                offset_b += coord * strides_b[d];                                         \
            }                                                                             \
                                                                                          \
            out[i] = OPERATION;                                                           \
        }                                                                                 \
    }
