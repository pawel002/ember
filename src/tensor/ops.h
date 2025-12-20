#ifndef OPS_H
#define OPS_H

#ifdef __cplusplus
extern "C" {
#endif

    void add(const float* a, const float* b, float* out, int size);
    void subtract(const float* a, const float* b, float* out, int size);

#ifdef __cplusplus
}
#endif

#endif