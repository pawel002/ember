/* CPU reference for the fused attention op (see flash_gpu.cu for the tiled CUDA
 * version and the layout description). This one is written for clarity, not
 * speed: it materializes one score row at a time, which is enough for the CPU
 * backend's tests. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "transformer.h"

int attention_supported(int dh)
{
    (void)dh;
    return 1;
}

void attention_fwd(const float *q, const float *k, const float *v, float *out, float *lse,
                   int batch, int sq, int sk, int dh, float scale, int causal)
{
    float *p = (float *)malloc((size_t)sk * sizeof(float));
    if (!p) return;

    for (int b = 0; b < batch; b++) {
        const float *qb = q + (size_t)b * sq * dh;
        const float *kb = k + (size_t)b * sk * dh;
        const float *vb = v + (size_t)b * sk * dh;
        float *ob = out + (size_t)b * sq * dh;
        float *lb = lse + (size_t)b * sq;

        for (int i = 0; i < sq; i++) {
            int limit = causal ? (i + 1 < sk ? i + 1 : sk) : sk;
            float m = -INFINITY;
            for (int j = 0; j < limit; j++) {
                float s = 0.0f;
                for (int d = 0; d < dh; d++)
                    s += qb[(size_t)i * dh + d] * kb[(size_t)j * dh + d];
                p[j] = s * scale;
                if (p[j] > m) m = p[j];
            }
            float l = 0.0f;
            for (int j = 0; j < limit; j++) {
                p[j] = expf(p[j] - m);
                l += p[j];
            }
            float inv = l > 0.0f ? 1.0f / l : 0.0f;
            for (int d = 0; d < dh; d++) {
                float acc = 0.0f;
                for (int j = 0; j < limit; j++) acc += p[j] * vb[(size_t)j * dh + d];
                ob[(size_t)i * dh + d] = acc * inv;
            }
            lb[i] = l > 0.0f ? (m + logf(l)) : -INFINITY;
        }
    }
    free(p);
}

void attention_bwd(const float *dout, const float *q, const float *k, const float *v,
                   const float *out, const float *lse, float *rowdot, float *dq, float *dk,
                   float *dv, int batch, int sq, int sk, int dh, float scale, int causal)
{
    memset(dq, 0, (size_t)batch * sq * dh * sizeof(float));
    memset(dk, 0, (size_t)batch * sk * dh * sizeof(float));
    memset(dv, 0, (size_t)batch * sk * dh * sizeof(float));

    float *p = (float *)malloc((size_t)sk * sizeof(float));
    float *ds = (float *)malloc((size_t)sk * sizeof(float));
    if (!p || !ds) {
        free(p);
        free(ds);
        return;
    }

    for (int b = 0; b < batch; b++) {
        const float *qb = q + (size_t)b * sq * dh;
        const float *kb = k + (size_t)b * sk * dh;
        const float *vb = v + (size_t)b * sk * dh;
        const float *gb = dout + (size_t)b * sq * dh;
        const float *ob = out + (size_t)b * sq * dh;
        const float *lb = lse + (size_t)b * sq;
        float *dqb = dq + (size_t)b * sq * dh;
        float *dkb = dk + (size_t)b * sk * dh;
        float *dvb = dv + (size_t)b * sk * dh;
        float *rd = rowdot + (size_t)b * sq;

        for (int i = 0; i < sq; i++) {
            int limit = causal ? (i + 1 < sk ? i + 1 : sk) : sk;

            float d_i = 0.0f;
            for (int d = 0; d < dh; d++) d_i += gb[(size_t)i * dh + d] * ob[(size_t)i * dh + d];
            rd[i] = d_i;

            for (int j = 0; j < limit; j++) {
                float s = 0.0f;
                for (int d = 0; d < dh; d++)
                    s += qb[(size_t)i * dh + d] * kb[(size_t)j * dh + d];
                p[j] = (lb[i] == -INFINITY) ? 0.0f : expf(s * scale - lb[i]);
            }

            for (int j = 0; j < limit; j++) {
                float dp = 0.0f;
                for (int d = 0; d < dh; d++)
                    dp += gb[(size_t)i * dh + d] * vb[(size_t)j * dh + d];
                ds[j] = p[j] * (dp - d_i);
                for (int d = 0; d < dh; d++)
                    dvb[(size_t)j * dh + d] += p[j] * gb[(size_t)i * dh + d];
            }

            for (int j = 0; j < limit; j++) {
                for (int d = 0; d < dh; d++) {
                    dqb[(size_t)i * dh + d] += scale * ds[j] * kb[(size_t)j * dh + d];
                    dkb[(size_t)j * dh + d] += scale * ds[j] * qb[(size_t)i * dh + d];
                }
            }
        }
    }
    free(p);
    free(ds);
}
