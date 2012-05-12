#ifndef NLFILTER_MATH_UTIL_H
#define NLFILTER_MATH_UTIL_H

#include <cutil_math.h>
#include <device_functions.h>

__device__ inline float3 saturate(float3 v)
{
    return make_float3(saturate(v.x),saturate(v.y),saturate(v.z));
}

__device__ inline float4 saturate(float4 v)
{
    return make_float4(saturate(v.x),saturate(v.y),saturate(v.z),saturate(v.w));
}

__device__ inline float3 rint(float3 v)
{
    return make_float3(rint(v.x),rint(v.y),rint(v.z));
}

__device__ inline float2 floor(float2 v)
{
    return make_float2(floor(v.x),floor(v.y));
}

__device__ inline float3 pow(float3 v,float n)
{
    return make_float3(pow(v.x,n),pow(v.y,n),pow(v.z,n));
}

__device__ inline float3 sqrt(float3 v)
{
    return make_float3(sqrt(v.x),sqrt(v.y),sqrt(v.z));
}

#endif
