#include <math.h>
#include <string.h>

#include "transformer.h"

/* CPU reference implementations of the transformer building blocks. Same math
 * as transformer_gpu.cu; used when Ember is built without a CUDA toolkit. */

/* ================= softmax ================= */
void softmax_axis(const float *x, float *out, int outer, int inner, int axis_dim)
{
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            int base = o * (axis_dim * inner) + i;
            float m = x[base];
            for (int r = 1; r < axis_dim; r++)
                if (x[base + r * inner] > m) m = x[base + r * inner];
            float s = 0.0f;
            for (int r = 0; r < axis_dim; r++) {
                float e = expf(x[base + r * inner] - m);
                out[base + r * inner] = e;
                s += e;
            }
            float inv = 1.0f / s;
            for (int r = 0; r < axis_dim; r++) out[base + r * inner] *= inv;
        }
    }
}

void softmax_axis_bwd(const float *dout, const float *y, float *dx, int outer, int inner,
                      int axis_dim)
{
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            int base = o * (axis_dim * inner) + i;
            float dot = 0.0f;
            for (int r = 0; r < axis_dim; r++) dot += dout[base + r * inner] * y[base + r * inner];
            for (int r = 0; r < axis_dim; r++) {
                int p = base + r * inner;
                dx[p] = y[p] * (dout[p] - dot);
            }
        }
    }
}

static void softmax_rows_impl(const float *x, float *out, int rows, int D, int causal, int sq)
{
    for (int row = 0; row < rows; row++) {
        const float *xr = x + (size_t)row * D;
        float *outr = out + (size_t)row * D;
        int limit = causal ? (row % sq) + 1 : D;

        float m = xr[0];
        for (int j = 1; j < limit; j++)
            if (xr[j] > m) m = xr[j];
        float s = 0.0f;
        for (int j = 0; j < limit; j++) s += expf(xr[j] - m);
        float inv = 1.0f / s;
        for (int j = 0; j < D; j++) outr[j] = (j < limit) ? expf(xr[j] - m) * inv : 0.0f;
    }
}

void softmax_rows(const float *x, float *out, int rows, int D)
{
    softmax_rows_impl(x, out, rows, D, 0, 0);
}

void softmax_rows_causal(const float *x, float *out, int rows, int D, int sq)
{
    softmax_rows_impl(x, out, rows, D, 1, sq);
}

/* ================= LayerNorm ================= */
void layernorm_fwd(const float *x, const float *gamma, const float *beta, float *out, float *mean,
                   float *rstd, int N, int D, float eps)
{
    for (int row = 0; row < N; row++) {
        const float *xr = x + (size_t)row * D;
        float *outr = out + (size_t)row * D;

        float s = 0.0f;
        for (int j = 0; j < D; j++) s += xr[j];
        float mu = s / D;

        float vs = 0.0f;
        for (int j = 0; j < D; j++) {
            float d = xr[j] - mu;
            vs += d * d;
        }
        float rs = 1.0f / sqrtf(vs / D + eps);

        mean[row] = mu;
        rstd[row] = rs;
        for (int j = 0; j < D; j++) outr[j] = (xr[j] - mu) * rs * gamma[j] + beta[j];
    }
}

void layernorm_bwd(const float *dout, const float *x, const float *gamma, const float *mean,
                   const float *rstd, float *dx, float *dgamma, float *dbeta, int N, int D)
{
    memset(dgamma, 0, (size_t)D * sizeof(float));
    memset(dbeta, 0, (size_t)D * sizeof(float));

    for (int row = 0; row < N; row++) {
        const float *dor = dout + (size_t)row * D;
        const float *xr = x + (size_t)row * D;
        float *dxr = dx + (size_t)row * D;
        float mu = mean[row], rs = rstd[row];

        float c1 = 0.0f, c2 = 0.0f;
        for (int j = 0; j < D; j++) {
            float xhat = (xr[j] - mu) * rs;
            float dxh = dor[j] * gamma[j];
            c1 += dxh;
            c2 += dxh * xhat;
        }
        c1 /= D;
        c2 /= D;
        for (int j = 0; j < D; j++) {
            float xhat = (xr[j] - mu) * rs;
            float dxh = dor[j] * gamma[j];
            dxr[j] = rs * (dxh - c1 - xhat * c2);
            dgamma[j] += dor[j] * xhat;
            dbeta[j] += dor[j];
        }
    }
}

/* ================= heads permute ================= */
void permute_0213(const float *x, float *out, int d0, int d1, int d2, int d3)
{
    int total = d0 * d1 * d2 * d3;
    for (int idx = 0; idx < total; idx++) {
        int e = idx % d3;
        int r = idx / d3;
        int c = r % d2;
        r /= d2;
        int b = r % d1;
        int a = r / d1;
        int out_idx = ((a * d2 + c) * d1 + b) * d3 + e;
        out[out_idx] = x[idx];
    }
}

/* ================= batched GEMM ================= */
void bmm(const float *a, const float *b, float *out, int batch, int n, int m, int k, int transA,
         int transB, float alpha)
{
    for (int bt = 0; bt < batch; bt++) {
        const float *A = a + (size_t)bt * n * k;
        const float *B = b + (size_t)bt * k * m;
        float *C = out + (size_t)bt * n * m;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                float acc = 0.0f;
                for (int l = 0; l < k; l++) {
                    float av = transA ? A[l * n + i] : A[i * k + l];
                    float bv = transB ? B[j * k + l] : B[l * m + j];
                    acc += av * bv;
                }
                C[i * m + j] = alpha * acc;
            }
        }
    }
}

/* ================= embedding ================= */
void embedding_fwd(const float *weight, const int *idx, float *out, int n_idx, int dim)
{
    for (int t = 0; t < n_idx; t++)
        for (int d = 0; d < dim; d++) out[t * dim + d] = weight[(size_t)idx[t] * dim + d];
}

void embedding_bwd(const float *dout, const int *idx, float *dweight, int n_idx, int dim, int vocab)
{
    memset(dweight, 0, (size_t)vocab * dim * sizeof(float));
    for (int t = 0; t < n_idx; t++)
        for (int d = 0; d < dim; d++) dweight[(size_t)idx[t] * dim + d] += dout[t * dim + d];
}
