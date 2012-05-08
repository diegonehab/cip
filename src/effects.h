#ifndef NLFILTER_EFFECTS_H
#define NLFILTER_EFFECTS_H

#include "math_util.h"

// posterize ------------------------------------------------------------

template <class T>
__device__ inline T posterize(T v, int levels)
{
    return saturate(rint(v*(levels-1))/(levels-1));
}

// threshold ------------------------------------------------------------

__device__ inline float threshold(float v, float a)
{
    return v < a ? 0 : 1;
}

__device__ inline float3 threshold(float3 v, float a)
{
    return make_float3(threshold(v.x, a), threshold(v.y, a), threshold(v.z, a));
}

// grayscale  ------------------------------------------------------------

__device__ inline float grayscale(float3 in)
{
    return 0.2126f * in.x + 0.7152f*in.y + 0.0722f*in.z;
}

// scale -------------------------------------------------------------

template <class T>
__device__ inline T scale(T v, float a)
{
    return v*a;
}

// bias -------------------------------------------------------------

template <class T>
__device__ inline T bias(T v, float a)
{
    return v+a;
}

// replacement -------------------------------------------------------------

__device__ inline float replacement(float v, float3 old, float3 new_, float3 tau)
{
    float a = fabs(v-grayscale(old));

    if(a <= tau.x && a <= tau.y && a <= tau.z)
        return saturate(grayscale(new_)+a);
    else
        return v;
}

__device__ inline float3 replacement(float3 v, float3 old, float3 new_, float3 tau)
{
    float3 a = fabs(v-old);

    if(a.x <= tau.x && a.y <= tau.y && a.z <= tau.z)
        return saturate(new_+a);
    else
        return v;
}

// polynomial ---------------------------------------------------------------

__device__ inline float3 polynomial(float3 v, int N, float *coeff)
{
    float3 cur = make_float3(0,0,0),
           power = make_float3(1,1,1);
#pragma unroll
    for(int i=0; i<N; ++i)
    {
        cur += coeff[i]*power;
        power *= v;
    }
    return saturate(cur);
}

// root --------------------------------------------------------------------

__device__ inline float root(float v, float n)
{
    if(v < 0 || v < 0 || v < 0)
        v = 0;

    return saturate(pow(v,1/n));
}

__device__ inline float3 root(float3 v, float n)
{
    if(v.x < 0 || v.y < 0 || v.z < 0)
        v = make_float3(0,0,0);

    return saturate(pow(v,1/n));
}

// gradient_edge_detection ------------------------------------------------

template <class T>
__device__ inline T gradient_edge_detection(T dx, T dy)
{
    return saturate(sqrt(dx*dx+dy*dy));
}

// uniform_quantization ----------------------------------------------------

#if 0
__device__ inline float3 uniform_quatization(float v, int N)
{
    float dist = 1000.0f;
    float3 cur = make_float3(0,0,0);

#pragma unroll
    for(int i=0; i<N; ++i)
    {
        float3 a = fabs(
}
#endif

#endif
