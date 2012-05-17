#ifndef NLFILTER_BSPLINE3_SAMPLER_H
#define NLFILTER_BSPLINE3_SAMPLER_H

#include "math_util.h"

#define USE_FAST_CUBIC_INTERPOLATION 1

template <class T>
__device__ void bspline3_weights(T alpha, T &w0, T &w1, T &w2, T &w3,
                                 int k=0)
{
    T one_alpha = 1.0f - alpha,
      alpha2 = alpha * alpha,
      one_alpha2 = one_alpha*one_alpha;

    switch(k)
    {
    case 0:
        w0 = (1/6.f) * one_alpha2*one_alpha,
        w1 = (2/3.f) - 0.5f*alpha2*(2.0f-alpha),
        w2 = (2/3.f) - 0.5f*one_alpha2*(2.0f-one_alpha),
        w3 = (1/6.f) * alpha*alpha2;
        break;
    case 1:
        w0 = -0.5f*alpha2 + alpha - 0.5f;
        w1 = 1.5f*alpha2 - 2.0f*alpha;
        w2 = -1.5f*alpha2 + alpha+0.5f;
        w3 = 0.5f*alpha2;
        break;
    case 2:
        w0 = 1.0f - alpha;
        w1 = 3.0f*alpha - 2.0f;
        w2 = -3.0f*alpha+1.0f;
        w3 = alpha;
        break;
    }
}

template <class S>
class bspline3_sampler
{
public:
    typedef S sampler_type;
    typedef typename S::result_type result_type;

    __device__ inline 
    result_type operator()(float2 pos, int kx=0, int ky=0) const
    {
        S sampler;

        pos -= 0.5f;

#if USE_FAST_CUBIC_INTERPOLATION
        if(kx < 2 && ky < 2)
        {
            float2 index = floor(pos);
            float2 alpha = pos - index;

            float2 w[4];

            bspline3_weights(alpha.x, w[0].x, w[1].x, w[2].x, w[3].x, kx);
            bspline3_weights(alpha.y, w[0].y, w[1].y, w[2].y, w[3].y, ky);


            float2 g0 = w[0] + w[1],
                   g1 = w[2] + w[3],
                  // h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
                   h0 = (w[1] / g0) - 0.5f + index,
                   h1 = (w[3] / g1) + 1.5f + index;

            // fetch the four linear
            result_type tex00 = sampler(h0.x, h0.y),
                        tex10 = sampler(h1.x, h0.y),
                        tex01 = sampler(h0.x, h1.y),
                        tex11 = sampler(h1.x, h1.y);

            if(ky == 0)
            {
                tex00 = lerp(tex01, tex00, g0.y);
                tex10 = lerp(tex11, tex10, g0.y);
            }
            else
            {
                tex00 = (tex00 - tex01)*g0.y;
                tex10 = (tex10 - tex11)*g0.y;
            }

            if(kx == 0)
                return lerp(tex10, tex00, g0.x);
            else
                return (tex00 - tex10)*g0.x;
        }
        else if(kx == 2 && ky < 2)
        {
            pos.x += 0.5f;
            float index = floor(pos.y);
            float alpha = pos.y - index;

            float w0, w1, w2, w3;

            bspline3_weights(alpha, w0, w1, w2, w3, ky);

            float g0 = w0 + w1,
                  g1 = w2 + w3,
                  // h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
                  h0 = (w1 / g0) - 0.5f + index,
                  h1 = (w3 / g1) + 1.5f + index;

            // fetch the six linear
            result_type tex00 = sampler(pos.x-1, h0),
                        tex10 = sampler(pos.x, h0),
                        tex20 = sampler(pos.x+1, h0),

                        tex01 = sampler(pos.x-1, h1),
                        tex11 = sampler(pos.x, h1),
                        tex21 = sampler(pos.x+1, h1);

            // weigh along the y-direction
            if(ky == 0)
            {
                tex00 = lerp(tex01, tex00, g0);
                tex10 = lerp(tex11, tex10, g0);
                tex20 = lerp(tex21, tex20, g0);
            }
            else
            {
                tex00 = (tex00 - tex01)*g0;
                tex10 = (tex10 - tex11)*g0;
                tex20 = (tex20 - tex21)*g0;
            }

            // weigh along the x-direction
            return tex00-2*tex10+tex20;
        }
        else if(ky == 2 && kx < 2)
        {
            pos.y += 0.5f;
            float index = floor(pos.x);
            float alpha = pos.x - index;

            float w0, w1, w2, w3;

            bspline3_weights(alpha, w0, w1, w2, w3, kx);

            float g0 = w0 + w1,
                  g1 = w2 + w3,
                  // h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
                  h0 = (w1 / g0) - 0.5f + index,
                  h1 = (w3 / g1) + 1.5f + index;

            // fetch the six linear
            result_type tex00 = sampler(h0, pos.y-1),
                        tex01 = sampler(h0, pos.y ),
                        tex02 = sampler(h0, pos.y+1),

                        tex10 = sampler(h1, pos.y-1),
                        tex11 = sampler(h1, pos.y),
                        tex12 = sampler(h1, pos.y+1);

            // weigh along the y-direction
            tex00 = tex00-2*tex01+tex02;
            tex10 = tex10-2*tex11+tex12;

            // weigh along the x-direction
            if(kx == 0)
                return lerp(tex10, tex00, g0);
            else
                return (tex10 - tex00)*g0;
        }
        else
        {
            result_type tex00 = sampler(pos.x-1, pos.y-1),
                        tex01 = sampler(pos.x-1, pos.y ),
                        tex02 = sampler(pos.x-1, pos.y+1),

                        tex10 = sampler(pos.x, pos.y-1),
                        tex11 = sampler(pos.x, pos.y),
                        tex12 = sampler(pos.x, pos.y+1),

                        tex20 = sampler(pos.x+1, pos.y-1),
                        tex21 = sampler(pos.x+1, pos.y),
                        tex22 = sampler(pos.x+1, pos.y+1);

            // weigh along the y-direction
            tex00 = tex00-2*tex01+tex02;
            tex10 = tex10-2*tex11+tex12;
            tex20 = tex20-2*tex21+tex22;

            // weigh along the x-direction
            return tex00-2*tex10+tex20;
        }
#else
        float2 index = floor(pos);
        float2 alpha = pos - index;

        float2 w[4];

        bspline3_weights(alpha.x, w[0].x, w[1].x, w[2].x, w[3].x, kx);
        bspline3_weights(alpha.y, w[0].y, w[1].y, w[2].y, w[3].y, ky);

        result_type value =  cuda_traits<result_type>::make(0);

#pragma unroll

        for(int i=0; i<4; ++i)
        {
#pragma unroll
            for(int j=0; j<4; ++j)
                value += sampler(index.x+0.5f+j-1,index.y+0.5f+i-1)*w[i].y*w[j].x;
        }
        return value;
#endif
    }
};

#endif
