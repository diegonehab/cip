#include <cutil.h>
#include "filter.h"
#include "effects.h"
#include "symbol.h"
#include "image_util.h"
#include "blue_noise.h"
#include "bspline3_sampler.h"

#define USE_LAUNCH_BOUNDS 1
const int BW_F1 = 32, // cuda block width
#if SAMPDIM == 8
          BH_F1 = 8; 
#else
          BH_F1 = 16;
#endif

const int BW_F2 = 32,
          BH_F2 = 8; 

#if USE_LAUNCH_BOUNDS
const int 
#if SAMPDIM == 8
          NB_F1 = 2,  // number of blocks resident per SM
#else
          NB_F1 = 1,  // number of blocks resident per SM
#endif
          NB_F2 = 4;
#endif


__constant__ float2 blue_noise[SAMPDIM];
__constant__ float bspline3_data[SAMPDIM*KS*KS];

__constant__ filter_operation filter_op;


// do the actual value processing according to what's in 'filter_op'
template <class S>
__device__ typename S::result_type do_filter(const S &sampler, float2 pos)
{
    switch(filter_op.type)
    {
    case EFFECT_POSTERIZE:
        return posterize(sampler(pos), filter_op.levels);
    case EFFECT_SCALE:
        return scale(sampler(pos),filter_op.scale);
    case EFFECT_BIAS:
        return bias(sampler(pos),filter_op.bias);
    case EFFECT_ROOT:
        return root(sampler(pos),filter_op.degree);
    case EFFECT_THRESHOLD:
        return threshold(sampler(pos),filter_op.threshold);
    case EFFECT_REPLACEMENT:
        return replacement(sampler(pos), 
                           filter_op.old_color, 
                           filter_op.new_color, 
                           filter_op.tau);
    case EFFECT_GRADIENT_EDGE_DETECTION:
        return gradient_edge_detection(sampler(pos,1,0),sampler(pos,0,1));
    case EFFECT_IDENTITY:
    default:
        return sampler(pos);
    }
}

//{{{ Grayscale filtering ===================================================

texture<float, 2, cudaReadModeElementType> t_in_gray;

__device__ float texfetch_gray(float x, float y)
{
    return tex2D(t_in_gray, x, y);
}

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW_F1*BH_F1, NB_F1)
#endif
void filter_kernel1(float2 *out,/*{{{*/
                    int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW_F1+tx, y = blockIdx.y*BH_F1+ty;

    if(x >= width || y >= height)
        return;

    int imgstride = rowstride*height;

    // output will point to the pixel we're processing now
    int idx = y*rowstride+x;
    out += idx;

    // we're using some smem as registers not to blow up the register space,
    // here we define how much 'registers' are in smem, the rest is used
    // in regular registers
    
    const int SMEM_SIZE = 3,
              REG_SIZE = KS*KS-SMEM_SIZE;

    __shared__ float2 _sum[BH_F1][SMEM_SIZE][BW_F1];
    float2 (*ssum)[BW_F1] = (float2 (*)[BW_F1]) &_sum[ty][0][tx];

    float2 sum[REG_SIZE];

    // Init registers to zero
    for(int i=0; i<REG_SIZE; ++i)
        sum[i] = make_float2(0,0);

#pragma unroll
    for(int i=0; i<SMEM_SIZE; ++i)
        *ssum[i] = make_float2(0,0);

    // top-left position of the kernel support
    float2 p = make_float2(x,y)-1.5f;

    float *bspline3 = bspline3_data;

    bspline3_sampler<float> sampler(texfetch_gray);

    for(int s=0; s<SAMPDIM; ++s)
    {
        float value = do_filter(sampler, p+blue_noise[s]);

        // scans through the kernel support, collecting data for each position
#pragma unroll
        for(int i=0; i<SMEM_SIZE; ++i)
        {
            float wij = bspline3[i];

            *ssum[i] += make_float2(value*wij, wij);
        }
        bspline3 += SMEM_SIZE;
#pragma unroll
        for(int i=0; i<REG_SIZE; ++i)
        {
            float wij = bspline3[i];

            sum[i] += make_float2(value*wij, wij);
        }
        bspline3 += REG_SIZE;
    }

    // writes out to gmem what's in the registers
#pragma unroll
    for(int i=0; i<SMEM_SIZE; ++i)
    {
        *out = *ssum[i];
        out += imgstride;
    }

#pragma unroll
    for(int i=0; i<REG_SIZE; ++i)
    {
        *out = sum[i];
        out += imgstride;
    }
}/*}}}*/

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW_F2*BH_F2, NB_F2)
#endif
void filter_kernel2(float *out, /*{{{*/
                    const float2 *in,
                    int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW_F2+tx, y = blockIdx.y*BH_F2+ty;

    // out of bounds? goodbye
    if(x >= width || y >= height)
        return;

    // size of each image plane
    int imgstride = rowstride*height;

    // in and out points to the input/output pixel we're processing
    int idx = y*rowstride+x;
    in += idx;
    out += idx;

    // treat corner cases where the support is outside the image
    int mi = min(y+KS,height)-y,
        mj = min(x+KS,width)-x;

    // sum the contribution of nearby pixels
    float2 sum = make_float2(0,0);

    int drow = rowstride+imgstride*KS,
        dcol = imgstride+1;

#pragma unroll
    for(int i=0; i<mi; ++i)
    {
        const float2 *in_row = in;
#pragma unroll
        for(int j=0; j<mj; ++j)
        {
            sum += *in;

            in += dcol;
        }
        in = in_row + drow;
    }

    *out = sum.x/sum.y;
}/*}}}*/

void filter(dvector<float> &v, int width, int height, int rowstride,/*{{{*/
            const filter_operation &op)
{
    copy_to_symbol("filter_op",op);

    // copy the input data to a texture
    cudaArray *a_in;
    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
    cudaMallocArray(&a_in, &ccd, width, height);
    cudaMemcpy2DToArray(a_in, 0, 0, v, rowstride*sizeof(float),
                        width*sizeof(float), height,
                        cudaMemcpyDeviceToDevice);

    t_in_gray.normalized = false;
    t_in_gray.filterMode = cudaFilterModeLinear;

    t_in_gray.addressMode[0] = t_in_gray.addressMode[1] = cudaAddressModeClamp;

    cudaBindTextureToArray(t_in_gray, a_in);

    dvector<float2> c(v.size()*KS*KS);
                   
    {
        dim3 bdim(BW_F1,BH_F1),
             gdim((width+bdim.x-1)/bdim.x, (height+bdim.y-1)/bdim.y);
        filter_kernel1<<<gdim, bdim>>>(c, width, height,rowstride);
    }

    {
        dim3 bdim(BW_F2,BH_F2),
             gdim((width+bdim.x-1)/bdim.x, (height+bdim.y-1)/bdim.y);
        filter_kernel2<<<gdim, bdim>>>(v, c,
                                       width, height, rowstride);
    }

    cudaUnbindTexture(t_in_gray);
    cudaFreeArray(a_in);
}/*}}}*/
/*}}}*/

#if CUDA_SM >= 20

//{{{ RGB filtering =========================================================

texture<float4, 2, cudaReadModeElementType> t_in_rgba;

__device__ float3 texfetch_rgba(float x, float y)
{
    return make_float3(tex2D(t_in_rgba, x, y));
}

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW_F1*BH_F1, NB_F1)
#endif
void filter_kernel1(float *out_r, float *out_g, float *out_b, float *out_w,/*{{{*/
                    int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW_F1+tx, y = blockIdx.y*BH_F1+ty;

    if(x >= width || y >= height)
        return;

    int imgstride = rowstride*height;

    // output will point to the pixel we're processing now
    int idx = y*rowstride+x;
    out_r += idx;
    out_g += idx;
    out_b += idx;
    out_w += idx;

    // we're using some smem as registers not to blow up the register space,
    // here we define how much 'registers' are in smem, the rest is used
    // in regular registers
    const int SMEM_SIZE = 5,
              REG_SIZE = KS*KS-SMEM_SIZE;

    __shared__ float4 _sum[BH_F1][SMEM_SIZE][BW_F1];
    float4 (*ssum)[BW_F1] = (float4 (*)[BW_F1]) &_sum[ty][0][tx];

    // zeroes out the registers
    float4 sum[REG_SIZE] = {0};

#pragma unroll
    for(int i=0; i<SMEM_SIZE; ++i)
        *ssum[i] = make_float4(0,0,0,0);

    // top-left position of the kernel support
    float2 p = make_float2(x,y)-1.5f;

    float *bspline3 = bspline3_data;
    
    bspline3_sampler<float3> sampler(texfetch_rgba);

    for(int s=0; s<SAMPDIM; ++s)
    {
        float3 value = do_filter(sampler, p+blue_noise[s]);

        // scans through the kernel support, collecting data for each position
#pragma unroll
        for(int i=0; i<SMEM_SIZE; ++i)
        {
            float wij = bspline3[i];

            *ssum[i] += make_float4(value*wij, wij);
        }
        bspline3 += SMEM_SIZE;
#pragma unroll
        for(int i=0; i<REG_SIZE; ++i)
        {
            float wij = bspline3[i];

            sum[i] += make_float4(value*wij, wij);
        }
        bspline3 += REG_SIZE;
    }

    // writes out to gmem what's in the registers
#pragma unroll
    for(int i=0; i<SMEM_SIZE; ++i)
    {
        *out_r = ssum[i]->x;
        *out_g = ssum[i]->y;
        *out_b = ssum[i]->z;
        *out_w = ssum[i]->w;

        out_r += imgstride;
        out_g += imgstride;
        out_b += imgstride;
        out_w += imgstride;
    }

#pragma unroll
    for(int i=0; i<REG_SIZE; ++i)
    {
        *out_r = sum[i].x;
        *out_g = sum[i].y;
        *out_b = sum[i].z;
        *out_w = sum[i].w;

        out_r += imgstride;
        out_g += imgstride;
        out_b += imgstride;
        out_w += imgstride;
    }
}/*}}}*/

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW_F2*BH_F2, NB_F2)
#endif
void filter_kernel2(float *out_r, float *out_g, float *out_b, /*{{{*/
                    const float *in_r, const float *in_g,
                    const float *in_b, const float *in_w,
                    int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW_F2+tx, y = blockIdx.y*BH_F2+ty;

    // out of bounds? goodbye
    if(x >= width || y >= height)
        return;

    // size of each image plane
    int imgstride = rowstride*height;

    // in and out points to the input/output pixel we're processing
    int idx = y*rowstride+x;

    in_r += idx;
    in_g += idx;
    in_b += idx;
    in_w += idx;

    out_r += idx;
    out_g += idx;
    out_b += idx;

    // treat corner cases where the support is outside the image
    int mi = min(y+KS,height)-y,
        mj = min(x+KS,width)-x;

    // sum the contribution of nearby pixels
    float4 sum = make_float4(0,0,0,0);

    int drow = rowstride+imgstride*KS,
        dcol = imgstride+1;

#pragma unroll
    for(int i=0; i<mi; ++i)
    {
        const float *in_r_row = in_r,
                    *in_g_row = in_g,
                    *in_b_row = in_b,
                    *in_w_row = in_w;
#pragma unroll
        for(int j=0; j<mj; ++j)
        {
            sum.x += *in_r;
            sum.y += *in_g;
            sum.z += *in_b;
            sum.w += *in_w;

            in_r += dcol;
            in_g += dcol;
            in_b += dcol;
            in_w += dcol;
        }
        in_r = in_r_row + drow;
        in_g = in_g_row + drow;
        in_b = in_b_row + drow;
        in_w = in_w_row + drow;
    }


    *out_r = sum.x/sum.w;
    *out_g = sum.y/sum.w;
    *out_b = sum.z/sum.w;
}/*}}}*/

void filter(dvector<float> imgchan[3], int width, int height, int rowstride,/*{{{*/
            const filter_operation &op)
{
    copy_to_symbol("filter_op",op);

    dvector<float4> img(imgchan[0].size());

    compose(img, imgchan, width, height, rowstride);

    // copy the input data to a texture
    cudaArray *a_in;
    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&a_in, &ccd, width, height);
    cudaMemcpy2DToArray(a_in, 0, 0, img, rowstride*sizeof(float4),
                        width*sizeof(float4), height,
                        cudaMemcpyDeviceToDevice);

    t_in_rgba.normalized = false;
    t_in_rgba.filterMode = cudaFilterModeLinear;

    t_in_rgba.addressMode[0] = t_in_rgba.addressMode[1] = cudaAddressModeClamp;

    cudaBindTextureToArray(t_in_rgba, a_in);

    dvector<float> temp[4];
    for(int i=0; i<4; ++i)
        temp[i].resize(rowstride*height*KS*KS);
                   
    {
        dim3 bdim(BW_F1,BH_F1),
             gdim((width+bdim.x-1)/bdim.x, (height+bdim.y-1)/bdim.y);
        filter_kernel1<<<gdim, bdim>>>(temp[0], temp[1], temp[2], temp[3], 
                                       width, height, rowstride);
    }

    {
        dim3 bdim(BW_F2,BH_F2),
             gdim((width+bdim.x-1)/bdim.x, (height+bdim.y-1)/bdim.y);
        filter_kernel2<<<gdim, bdim>>>(imgchan[0], imgchan[1], imgchan[2], 
                                       temp[0], temp[1], temp[2], temp[3],
                                       width, height, rowstride);
    }

    cudaUnbindTexture(t_in_rgba);
    cudaFreeArray(a_in);
}/*}}}*/
/*}}}*/

#endif


float bspline3(float r)
{
    r = std::abs(r);

    if (r < 1.f) 
        return (4.f + r*r*(-6.f + 3.f*r))/6.f;
    else if (r < 2.f) 
        return  (8.f + r*(-12.f + (6.f - r)*r))/6.f;
    else 
        return 0.f;
}


void init_blue_noise()
{
    std::vector<float2> blue_noise;
    std::vector<float> bspline3_data;
    blue_noise.reserve(SAMPDIM);
    bspline3_data.reserve(SAMPDIM*KS*KS);
    for(int i=0; i<SAMPDIM; ++i)
    {
        float2 n = make_float2(blue_noise_x[i], blue_noise_y[i]);

        blue_noise.push_back(n);
        for(int y=0; y<KS; ++y)
        {
            for(int x=0; x<KS; ++x)
            {
                bspline3_data.push_back(bspline3(x+n.x-1.5)*
                                        bspline3(y+n.y-1.5)/SAMPDIM);
            }
        }
    }
    copy_to_symbol("blue_noise",blue_noise);
    copy_to_symbol("bspline3_data",bspline3_data);
}
