// Copyright 2012--2020 Leonardo Sacht, Rodolfo Schulz de Lima, Diego Nehab
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#ifndef NLFILTER_FILTER_H
#define NLFILTER_FILTER_H

#include "config.h"
#include "image_util.h"
#include "dimage.h"

// number of samples per pixel
#define SAMPDIM 64

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
    EFFECT_GRADIENT_EDGE_DETECTION,
    EFFECT_LAPLACE_EDGE_ENHANCEMENT,
    EFFECT_YAROSLAVSKY_BILATERAL,
    EFFECT_BRIGHTNESS_CONTRAST,
    EFFECT_HUE_SATURATION_LIGHTNESS,
    EFFECT_UNSHARP_MASK,
    EFFECT_BILATERAL,
    EFFECT_EMBOSS
};

enum filter_type
{
    FILTER_BSPLINE3,
    FILTER_CARDINAL_BSPLINE3,
    FILTER_MITCHELL_NETRAVALI,
    FILTER_BOX,
    FILTER_SACHT_NEHAB3
};

enum filter_flags
{
    VERBOSE=1
};


struct filter_operation
{
    effect_type type;

    filter_type pre_filter, post_filter;

    bool use_supersampling;

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
        struct
        {
            float minimum, maximum;
        };

        // replacement
        struct
        {
            float3 old_color, new_color, tau;
        };

        // laplace edge enhancement
        float multiple;

        // yaroslavsky bilateral
        struct
        {
            float rho, h;
        };

        // brightness and contrast
        struct
        {
            float brightness, contrast;
        };

        // hue, saturation and lightness
        struct
        {
            float hue, saturation, lightness;
        };

        // unsharp mask
        struct
        {
            float threshold,
                  amount, sigma;
        };

        // bilateral
        struct
        {
            float sigma_s, sigma_r;
        };

        // emboss
        struct
        {
            float offset,  
                  dummy2; // will use amount above
        };
    };
};

struct filter_plan;

template <int C> // C = number of channels
filter_plan *filter_create_plan(dimage_ptr<const float,C> src_img, 
                                const filter_operation &op, int flags=0);

void free(filter_plan *plan);

template <int C> // C = number of channels
void filter(filter_plan *plan, dimage_ptr<float, C> img, 
            const filter_operation &op);

#endif
