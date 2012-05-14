#ifndef NLFILTER_EFFECTS_H
#define NLFILTER_EFFECTS_H

#include "math_util.h"

// posterize ------------------------------------------------------------

template <class T>
__device__ inline 
T posterize(T v, int levels)
{
    return saturate(rint(v*(levels-1))/(levels-1));
}

// threshold ------------------------------------------------------------

__device__ inline 
float threshold(float v, float a)
{
    return v < a ? 0 : 1;
}

__device__ inline 
float3 threshold(float3 v, float a)
{
    return make_float3(threshold(v.x, a), threshold(v.y, a), threshold(v.z, a));
}

// grayscale  ------------------------------------------------------------

__device__ inline 
float grayscale(float3 in)
{
    return 0.2126f * in.x + 0.7152f*in.y + 0.0722f*in.z;
}

// scale -------------------------------------------------------------

template <class T>
__device__ inline 
T scale(T v, float a)
{
    return v*a;
}

// bias -------------------------------------------------------------

template <class T>
__device__ inline 
T bias(T v, float a)
{
    return v+a;
}

// replacement -------------------------------------------------------------

__device__ inline 
float replacement(float v, float3 old, float3 new_, float3 tau)
{
    float a = fabs(v-grayscale(old));

    if(a <= tau.x && a <= tau.y && a <= tau.z)
        return saturate(grayscale(new_)+a);
    else
        return v;
}

__device__ inline 
float3 replacement(float3 v, float3 old, float3 new_, float3 tau)
{
    float3 a = fabs(v-old);

    if(a.x <= tau.x && a.y <= tau.y && a.z <= tau.z)
        return saturate(new_+a);
    else
        return v;
}

// polynomial ---------------------------------------------------------------

__device__ inline 
float3 polynomial(float3 v, int N, float *coeff)
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

__device__ inline 
float root(float v, float n)
{
    if(v < 0 || v < 0 || v < 0)
        v = 0;

    return saturate(pow(v,1/n));
}

__device__ inline 
float3 root(float3 v, float n)
{
    if(v.x < 0 || v.y < 0 || v.z < 0)
        v = make_float3(0,0,0);

    return saturate(pow(v,1/n));
}

// gradient_edge_detection ------------------------------------------------

template <class T>
__device__ inline 
T gradient_edge_detection(T dx, T dy)
{
    return saturate(sqrt(dx*dx+dy*dy));
}

// laplacian  ----------------------------------------------------

template <class T>
__device__ inline 
T laplacian(T dxx, T dyy)
{
    return saturate(dxx+dyy);
}

// laplace_edge_enhacement -----------------------------------------


template <class T>
__device__ inline 
T laplace_edge_enhancement(T v, T dxx, T dyy, float multiple)
{
    return saturate(v - multiple*(dxx-dyy));
}

// yaroslavsky bilateral --------------------------------------------
namespace detail
{

__device__ inline
float calc_yb_g_tilde(float param, float E)
{
    if(param < 0.000001f)
        return 1.f/6;
    else
        return (param*exp(-(param*param)))/(3.f*E);
}


__device__ inline
float3 calc_yb_g_tilde(float3 param, float3 E)
{
    return make_float3(calc_yb_g_tilde(param.x,E.x),
                       calc_yb_g_tilde(param.y,E.y),
                       calc_yb_g_tilde(param.z,E.z));
}

__device__ inline
float calc_yb_f_tilde(float param, float g_tilde)
{
    if(param < 0.000001f)
        return 1.f/6;
    else
        return 3.f*g_tilde+((3.f*g_tilde-0.5f)/(param*param));
}

__device__ inline
float3 calc_yb_f_tilde(float3 param, float3 g_tilde)
{
    return make_float3(calc_yb_f_tilde(param.x,g_tilde.x),
                       calc_yb_f_tilde(param.y,g_tilde.y),
                       calc_yb_f_tilde(param.z,g_tilde.z));
}
}

template <class T>
__device__ inline
T yaroslavsky_bilateral(T v, T dx, T dy, T dxy, T dxx, T dyy,
                        float rho, float h)
{
    T grad = sqrt(dx*dx + dy*dy),
      ort = (1.f/(grad*grad))*(dx*dx*dxx + 2*dx*dy*dxy + dy*dy*dyy),
      tan = (1.f/(grad*grad))*(dy*dy*dxx - 2*dx*dy*dxy + dx*dx*dyy),
      param = grad*rho / h;

    const float sqrt_pi = 1.77245385;
    T E = 2*((sqrt_pi/2.f)*erff(param));

    T g_tilde = detail::calc_yb_g_tilde(param, E),
      f_tilde = detail::calc_yb_f_tilde(param, g_tilde);

    return saturate(v + rho*rho*(f_tilde*ort + g_tilde*tan));
}

// brightness and contrast ---------------------------------------------

template <class T>
__device__ inline
T brightness_contrast(T v, float brightness, float contrast)
{
    // although we accept brightness values in the range [-1;1], the
    // operation uses [-0.5;0.5], so let's remap it
    brightness *= 0.5;

    if(brightness < 0)
        v *= (1+brightness);
    else
        v += (1-v)*brightness;

    const float PI = 3.14159265359;

    float slant = tan((contrast+1)*PI/4);
    return saturate((v-0.5)*slant + 0.5);
}

#endif
