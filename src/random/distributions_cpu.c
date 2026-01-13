#include <stdlib.h>

void uniform(float low, float high, float *out, int size)
{
    float range = high - low;
    for (int i = 0; i < size; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        out[i] = low + r * range;
    }
}
