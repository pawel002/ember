#include <stdlib.h>

#include "../core/memory.h"
#include "distributions.h"

/* Random values are drawn on the host (so seeding via the C stdlib `srand`
 * behaves identically on both backends) and copied to the device buffer. */

extern "C" {

void uniform(float low, float high, float *out, int size)
{
    float range = high - low;
    float *host = (float *)malloc((size_t)size * sizeof(float));

    for (int i = 0; i < size; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        host[i] = low + r * range;
    }

    copy_to_device(out, host, (size_t)size * sizeof(float));
    free(host);
}

void constant(float value, float *out, int size)
{
    float *host = (float *)malloc((size_t)size * sizeof(float));
    for (int i = 0; i < size; i++) host[i] = value;

    copy_to_device(out, host, (size_t)size * sizeof(float));
    free(host);
}
}
