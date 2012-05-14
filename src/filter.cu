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
template <effect_type OP, class S>
__device__ typename S::result_type do_filter(const S &sampler, float2 pos)
{
    switch(OP)
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
    case EFFECT_LAPLACIAN:
        return laplacian(sampler(pos,2,0),sampler(pos,0,2));
    case EFFECT_LAPLACE_EDGE_ENHANCEMENT:
        return laplace_edge_enhancement(sampler(pos),
                                        sampler(pos,2,0),sampler(pos,0,2),
                                        filter_op.multiple);
    case EFFECT_YAROSLAVSKY_BILATERAL:
        return yaroslavsky_bilateral(sampler(pos),
                                     sampler(pos,1,0), sampler(pos,0,1),
                                     sampler(pos,1,1),
                                     sampler(pos,2,0),sampler(pos,0,2),
                                     filter_op.rho, filter_op.h);
    case EFFECT_BRIGHTNESS_CONTRAST:
        return brightness_contrast(sampler(pos),filter_op.brightness,
                                   filter_op.contrast);
    case EFFECT_IDENTITY:
    default:
        return sampler(pos);
    }
}

template <int C>
struct filter_traits {};

template <int C>
struct sum_traits
    : pixel_traits<float,C+1>
{
    typedef typename pixel_traits<float,C+1>::pixel_type type;
};


template <effect_type OP,int C>
__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW_F1*BH_F1, NB_F1)
#endif
void filter_kernel1(dimage_ptr<typename sum_traits<C>::type,KS*KS> out)/*{{{*/
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW_F1+tx, y = blockIdx.y*BH_F1+ty;

    if(!out.is_inside(x,y))
        return;

    // output will point to the pixel we're processing now
    int idx = out.offset_at(x,y);
    out += idx;

    // we're using some smem as registers not to blow up the register space,
    // here we define how much 'registers' are in smem, the rest is used
    // in regular registers
    
    typedef filter_traits<C> cfg;

    typedef typename sum_traits<C>::type sum_type;
    typedef typename pixel_traits<float,C>::pixel_type pixel_type;

    const int SMEM_SIZE = cfg::smem_size,
              REG_SIZE = KS*KS-SMEM_SIZE;

    __shared__ sum_type _sum[BH_F1][SMEM_SIZE][BW_F1];
    sum_type (*ssum)[BW_F1] = (sum_type (*)[BW_F1]) &_sum[ty][0][tx];

    sum_type sum[REG_SIZE];

    // Init registers to zero
    for(int i=0; i<REG_SIZE; ++i)
        sum[i] = sum_traits<C>::make_pixel(0);

#pragma unroll
    for(int i=0; i<SMEM_SIZE; ++i)
        *ssum[i] = sum_traits<C>::make_pixel(0);

    // top-left position of the kernel support
    float2 p = make_float2(x,y)-1.5f+0.5f;

    float *bspline3 = bspline3_data;

    bspline3_sampler<typename cfg::texfetch_type> sampler;

    for(int s=0; s<SAMPDIM; ++s)
    {
        pixel_type value = do_filter<OP>(sampler, p+blue_noise[s]);

        // scans through the kernel support, collecting data for each position
#pragma unroll
        for(int i=0; i<SMEM_SIZE; ++i)
        {
            float wij = bspline3[i];

            *ssum[i] += sum_traits<C>::make_pixel(value*wij, wij);
        }
        bspline3 += SMEM_SIZE;
#pragma unroll
        for(int i=0; i<REG_SIZE; ++i)
        {
            float wij = bspline3[i];

            sum[i] += sum_traits<C>::make_pixel(value*wij, wij);
        }
        bspline3 += REG_SIZE;
    }

    // writes out to gmem what's in the registers
#pragma unroll
    for(int i=0; i<SMEM_SIZE; ++i)
        *out[i] = *ssum[i];

#pragma unroll
    for(int i=0; i<REG_SIZE; ++i)
        *out[SMEM_SIZE+i] = sum[i];
}/*}}}*/

template <int C>
__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW_F2*BH_F2, NB_F2)
#endif
void filter_kernel2(dimage_ptr<float,C> out, /*{{{*/
                    dimage_ptr<const typename sum_traits<C>::type,KS*KS> in)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW_F2+tx, y = blockIdx.y*BH_F2+ty;

    // out of bounds? goodbye
    if(!in.is_inside(x,y))
        return;

    // in and out points to the input/output pixel we're processing
    int idx = in.offset_at(x,y);
    in += idx;
    out += idx;

    // treat corner cases where the support is outside the image
    int mi = min(y+KS,in.height())-y,
        mj = min(x+KS,in.width())-x;

    // sum the contribution of nearby pixels
    typename sum_traits<C>::type sum = sum_traits<C>::make_pixel(0);

#pragma unroll
    for(int i=0; i<mi; ++i)
    {
#pragma unroll
        for(int j=0; j<mj; ++j)
        {
            sum += *in[i*KS+j];
            ++in;
        }
        in += in.rowstride()-mj;
    }

    *out = filter_traits<C>::normalize_sum(sum);
}/*}}}*/

template <int C>
void filter(dimage_ptr<float,C> img, const filter_operation &op)/*{{{*/
{
    typedef filter_traits<C> cfg;
    typedef typename pixel_traits<float,C>::texel_type texel_type;
    typedef typename sum_traits<C>::type sum_type;

    // copy the input data to a texture
    cudaArray *a_in;
    cudaChannelFormatDesc ccd 
        = cudaCreateChannelDesc<texel_type>();

    cudaMallocArray(&a_in, &ccd, img.width(), img.height());

    cfg::copy_to_array(a_in, img);

    cfg::tex().normalized = false;
    cfg::tex().filterMode = cudaFilterModeLinear;

    cfg::tex().addressMode[0]= cfg::tex().addressMode[1] = cudaAddressModeClamp;

    cudaBindTextureToArray(cfg::tex(), a_in);

    copy_to_symbol("filter_op",op);

    dimage<sum_type, KS*KS> temp(img.width(), img.height());

    dim3 bdim(BW_F1,BH_F1),
         gdim((img.width()+bdim.x-1)/bdim.x, (img.height()+bdim.y-1)/bdim.y);

    switch(op.type)
    {
    case EFFECT_IDENTITY:
        filter_kernel1<EFFECT_IDENTITY,C><<<gdim, bdim>>>(&temp);
        break;
    case EFFECT_POSTERIZE:
        filter_kernel1<EFFECT_POSTERIZE,C><<<gdim, bdim>>>(&temp);
        break;
    case EFFECT_SCALE:
        filter_kernel1<EFFECT_SCALE,C><<<gdim, bdim>>>(&temp);
        break;
    case EFFECT_BIAS:
        filter_kernel1<EFFECT_BIAS,C><<<gdim, bdim>>>(&temp);
        break;
    case EFFECT_ROOT:
        filter_kernel1<EFFECT_ROOT,C><<<gdim, bdim>>>(&temp);
        break;
    case EFFECT_THRESHOLD:
        filter_kernel1<EFFECT_THRESHOLD,C><<<gdim, bdim>>>(&temp);
        break;
    case EFFECT_REPLACEMENT:
        filter_kernel1<EFFECT_REPLACEMENT,C><<<gdim, bdim>>>(&temp);
        break;
    case EFFECT_GRADIENT_EDGE_DETECTION:
        filter_kernel1<EFFECT_GRADIENT_EDGE_DETECTION,C><<<gdim, bdim>>>(&temp);
        break;
    case EFFECT_LAPLACIAN:
        filter_kernel1<EFFECT_LAPLACIAN,C><<<gdim, bdim>>>(&temp);
        break;
    case EFFECT_LAPLACE_EDGE_ENHANCEMENT:
        filter_kernel1<EFFECT_LAPLACE_EDGE_ENHANCEMENT,C><<<gdim, bdim>>>(&temp);
        break;
    case EFFECT_YAROSLAVSKY_BILATERAL:
        filter_kernel1<EFFECT_YAROSLAVSKY_BILATERAL,C><<<gdim, bdim>>>(&temp);
        break;
    case EFFECT_BRIGHTNESS_CONTRAST:
        filter_kernel1<EFFECT_BRIGHTNESS_CONTRAST,C><<<gdim, bdim>>>(&temp);
        break;
    default:
        assert(false);
    }
                   
    {
        dim3 bdim(BW_F2,BH_F2),
             gdim((img.width()+bdim.x-1)/bdim.x,(img.height()+bdim.y-1)/bdim.y);
        filter_kernel2<C><<<gdim, bdim>>>(img, &temp);
    }

    cudaUnbindTexture(cfg::tex());
    cudaFreeArray(a_in);
}/*}}}*/


// Grayscale filtering ===================================================/*{{{*/

texture<float, 2, cudaReadModeElementType> t_in_gray;

struct texfetch_gray
{
    typedef float result_type;

    __device__ float operator()(float x, float y)
    {
        return tex2D(t_in_gray, x, y);
    }
};

template <> 
struct filter_traits<1>
{
    typedef texfetch_gray texfetch_type;
    static const int smem_size = 3;

    static 
    texture<float,2,cudaReadModeElementType> &tex() { return t_in_gray; }

    static void copy_to_array(cudaArray *out, dimage_ptr<float> in)
    {
        cudaMemcpy2DToArray(out, 0, 0, in, 
                            in.rowstride()*sizeof(float),
                            in.width()*sizeof(float), in.height(),
                            cudaMemcpyDeviceToDevice);
    }

    __device__ static float sample(float x, float y)
    {
        return tex2D(t_in_gray, x, y);
    }

    __device__ static float normalize_sum(float2 sum)
    {
        return sum.x / sum.y;
    }
};

template 
void filter(dimage_ptr<float,1> img, const filter_operation &op);
/*}}}*/

//{{{ RGB filtering =========================================================

#if CUDA_SM < 20
template<> 
void filter(dimage_ptr<float,3> img, const filter_operation &op)
{
    for(int i=0; i<3; ++i)
        filter(img[i], op);
}
#else

texture<float4, 2, cudaReadModeElementType> t_in_rgba;

struct texfetch_rgba
{
    typedef float3 result_type;

    __device__ float3 operator()(float x, float y)
    {
        return make_float3(tex2D(t_in_rgba, x, y));
    }
};

template <> 
struct filter_traits<3>
{
    typedef texfetch_rgba texfetch_type;
    static const int smem_size = 5;

    static texture<float4,2,cudaReadModeElementType> &tex() 
        { return t_in_rgba; }

    static void copy_to_array(cudaArray *out, dimage_ptr<float,3> img)
    {
        dimage<float3> temp;
        temp.resize(img.width(), img.height());
        convert(&temp, img);

        cudaMemcpy2DToArray(out, 0, 0, temp, 
                            temp.rowstride()*sizeof(float4),
                            temp.width()*sizeof(float4), temp.height(),
                            cudaMemcpyDeviceToDevice);
    }

    __device__ static float3 normalize_sum(float4 sum)
    {
        return make_float3(sum) / sum.w;
    }
};

template 
void filter(dimage_ptr<float,3> img, const filter_operation &op);

#endif
/*}}}*/

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
