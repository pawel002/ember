#ifndef OPS_H
#define OPS_H

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * Launch the Element-wise Addition Kernel on GPU.
     * 
     * Operation: out[i] = a[i] + b[i]
     * 
     * @param a    Pointer to first input array on GPU
     * @param b    Pointer to second input array on GPU
     * @param out  Pointer to result array on GPU (must be allocated)
     * @param size Number of elements
     */
    void launch_add(const float* a, const float* b, float* out, int size);

#ifdef __cplusplus
}
#endif

#endif