#include "math_util.h"
#include "image_util.h"

#define USE_LAUNCH_BOUNDS 1

const int BW = 32, // cuda block width
          BH = 6, // cuda block height
#if USE_SM20
          NB = 8;
#else
          NB = 5;
#endif

// compose -----------------------------------------------------

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void compose(float4 *output, 
             const float *r, const float *g, const float *b,
             int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(x >= width || y >= height)
        return;

    int idx = y*rowstride+x;

    output[idx] = make_float4(r[idx],g[idx],b[idx],0);
}

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void compose(uchar4 *output, 
             const float *r, const float *g, const float *b,
             int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(x >= width || y >= height)
        return;

    int idx = y*rowstride+x;

    float3 v = saturate(make_float3(r[idx],g[idx],b[idx]))*255.0f;
    output[idx] = make_uchar4(v.x,v.y,v.z,1);
}

template <class T>
void compose(dvector<T> &out, const dvector<float> in[3], 
             int width, int height, int rowstride)
{
    out.resize(rowstride*height);

    dim3 bdim(BW,BH),
         gdim((width+bdim.x-1)/bdim.x, (height+bdim.y-1)/bdim.y);

    compose<<<gdim, bdim>>>(out, in[0],in[1],in[2], width, height, rowstride);
}

template 
void compose(dvector<float4> &out, const dvector<float> in[3], 
             int width, int height, int rowstride);

template 
void compose(dvector<uchar4> &out, const dvector<float> in[3], 
             int width, int height, int rowstride);


// decompose -----------------------------------------------------

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void decompose(float *r, float *g, float *b, const float4 *input,
                          int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(x >= width || y >= height)
        return;

    int idx = y*rowstride+x;

    float4 v = input[idx];
    r[idx] = v.x;
    g[idx] = v.y;
    b[idx] = v.z;
}

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void decompose(float *r, float *g, float *b, const uchar4 *input,
                          int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(x >= width || y >= height)
        return;

    int idx = y*rowstride+x;

    uchar4 v = input[idx];
    r[idx] = v.x/255.0f;
    g[idx] = v.y/255.0f;
    b[idx] = v.z/255.0f;
}

template <class T>
void decompose(dvector<float> out[3], const dvector<T> &in, 
               int width, int height, int rowstride)
{
    for(int i=0; i<3; ++i)
        out[i].resize(rowstride*height);

    dim3 bdim(BW,BH),
         gdim((width+bdim.x-1)/bdim.x, (height+bdim.y-1)/bdim.y);

    decompose<<<gdim, bdim>>>(out[0],out[1],out[2], in, width, height, rowstride);
}

template
void decompose(dvector<float> out[3], const dvector<float4> &in, 
               int width, int height, int rowstride);

template 
void decompose(dvector<float> out[3], const dvector<uchar4> &in, 
               int width, int height, int rowstride);

// grayscale ------------------------------------------------------------

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void grayscale(float *output, const uchar4 *input,
               int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(x >= width || y >= height)
        return;

    int idx = y*rowstride+x;

    uchar4 in = input[idx];
    output[idx] = in.x/255.0f*0.2126f + in.y/255.0f*0.7152f + in.z/255.0f*0.0722f;
}

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void grayscale(float *output, const float4 *input,
               int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(x >= width || y >= height)
        return;

    int idx = y*rowstride+x;

    float4 in = input[idx];
    output[idx] = in.x*0.2126f + in.y*0.7152f + in.z*0.0722f;
}

template <class T>
void grayscale(dvector<float> &out, const dvector<T> &in, 
               int width, int height, int rowstride)
{
    out.resize(rowstride*height);

    dim3 bdim(BW,BH),
         gdim((width+bdim.x-1)/bdim.x, (height+bdim.y-1)/bdim.y);

    grayscale<<<gdim, bdim>>>(&out, &in, width, height, rowstride);
}


template
void grayscale(dvector<float> &out, const dvector<float4> &in, 
               int width, int height, int rowstride);

template 
void grayscale(dvector<float> &out, const dvector<uchar4> &in, 
               int width, int height, int rowstride);
