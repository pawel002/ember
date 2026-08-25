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

/* Offset of batch-head `bat`'s first row; see the layout note in
 * transformer.h. */
static size_t fa_base(int bat, int nheads, int dh, int seq, int sseq)
{
    return (size_t)(bat / nheads) * seq * sseq + (size_t)(bat % nheads) * dh;
}

void attention_fwd(const float *q, const float *k, const float *v, float *out, float *lse,
                   int batch, int sq, int sk, int dh, float scale, int causal, int nheads, int sseq)
{
    float *p = (float *)malloc((size_t)sk * sizeof(float));
    if (!p) return;

    for (int b = 0; b < batch; b++) {
        const float *qb = q + fa_base(b, nheads, dh, sq, sseq);
        const float *kb = k + fa_base(b, nheads, dh, sk, sseq);
        const float *vb = v + fa_base(b, nheads, dh, sk, sseq);
        float *ob = out + fa_base(b, nheads, dh, sq, sseq);
        float *lb = lse + (size_t)b * sq;

        for (int i = 0; i < sq; i++) {
            int limit = causal ? (i + 1 < sk ? i + 1 : sk) : sk;
            float m = -INFINITY;
            for (int j = 0; j < limit; j++) {
                float s = 0.0f;
                for (int d = 0; d < dh; d++)
                    s += qb[(size_t)i * sseq + d] * kb[(size_t)j * sseq + d];
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
                for (int j = 0; j < limit; j++) acc += p[j] * vb[(size_t)j * sseq + d];
                ob[(size_t)i * sseq + d] = acc * inv;
            }
            lb[i] = l > 0.0f ? (m + logf(l)) : -INFINITY;
        }
    }
    free(p);
}

void attention_bwd(const float *dout, const float *q, const float *k, const float *v,
                   const float *out, const float *lse, float *rowdot, float *dq, float *dk,
                   float *dv, int batch, int sq, int sk, int dh, float scale, int causal,
                   int nheads, int sseq)
{
    size_t nb = (size_t)(batch / nheads);
    memset(dq, 0, nb * sq * sseq * sizeof(float));
    memset(dk, 0, nb * sk * sseq * sizeof(float));
    memset(dv, 0, nb * sk * sseq * sizeof(float));

    float *p = (float *)malloc((size_t)sk * sizeof(float));
    float *ds = (float *)malloc((size_t)sk * sizeof(float));
    if (!p || !ds) {
        free(p);
        free(ds);
        return;
    }

    for (int b = 0; b < batch; b++) {
        const float *qb = q + fa_base(b, nheads, dh, sq, sseq);
        const float *kb = k + fa_base(b, nheads, dh, sk, sseq);
        const float *vb = v + fa_base(b, nheads, dh, sk, sseq);
        const float *gb = dout + fa_base(b, nheads, dh, sq, sseq);
        const float *ob = out + fa_base(b, nheads, dh, sq, sseq);
        const float *lb = lse + (size_t)b * sq;
        float *dqb = dq + fa_base(b, nheads, dh, sq, sseq);
        float *dkb = dk + fa_base(b, nheads, dh, sk, sseq);
        float *dvb = dv + fa_base(b, nheads, dh, sk, sseq);
        float *rd = rowdot + (size_t)b * sq;

        for (int i = 0; i < sq; i++) {
            int limit = causal ? (i + 1 < sk ? i + 1 : sk) : sk;

            float d_i = 0.0f;
            for (int d = 0; d < dh; d++) d_i += gb[(size_t)i * sseq + d] * ob[(size_t)i * sseq + d];
            rd[i] = d_i;

            for (int j = 0; j < limit; j++) {
                float s = 0.0f;
                for (int d = 0; d < dh; d++)
                    s += qb[(size_t)i * sseq + d] * kb[(size_t)j * sseq + d];
                p[j] = (lb[i] == -INFINITY) ? 0.0f : expf(s * scale - lb[i]);
            }

            for (int j = 0; j < limit; j++) {
                float dp = 0.0f;
                for (int d = 0; d < dh; d++)
                    dp += gb[(size_t)i * sseq + d] * vb[(size_t)j * sseq + d];
                ds[j] = p[j] * (dp - d_i);
                for (int d = 0; d < dh; d++)
                    dvb[(size_t)j * sseq + d] += p[j] * gb[(size_t)i * sseq + d];
            }

            for (int j = 0; j < limit; j++) {
                for (int d = 0; d < dh; d++) {
                    dqb[(size_t)i * sseq + d] += scale * ds[j] * kb[(size_t)j * sseq + d];
                    dkb[(size_t)j * sseq + d] += scale * ds[j] * qb[(size_t)i * sseq + d];
                }
            }
        }
    }
    free(p);
    free(ds);
}
