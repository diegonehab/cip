#ifndef NLFILTER_FILTER_H
#define NLFILTER_FILTER_H

#include "config.h"
#include "image_util.h"
#include "dimage.h"

// number of samples per pixel
#define SAMPDIM 8

#define KS 4 // pre- and post-filter kernel support size

void init_blue_noise();

enum effect_type
{
    EFFECT_IDENTITY,
    EFFECT_POSTERIZE,
    EFFECT_SCALE,
    EFFECT_BIAS,
    EFFECT_ROOT,
    EFFECT_THRESHOLD,
    EFFECT_REPLACEMENT,
    EFFECT_LAPLACIAN,
    EFFECT_GRADIENT_EDGE_DETECTION
};

struct filter_operation
{
    effect_type type;

    union
    {
        // posterize
        int levels;

        // scale
        float scale;

        // bias
        float bias;

        // root
        float degree;

        // threshold
        float threshold;

        // replacement
        struct
        {
            float3 old_color, new_color, tau;
        };
    };
};


template <int C> // C = number of channels
void filter(dimage_ptr<float, C> img, const filter_operation &op);

#endif
