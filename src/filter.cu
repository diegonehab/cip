#include <cutil.h>
#include "filter.h"
#include "timer.h"
#include "effects.h"
#include "symbol.h"
#include "image_util.h"
#include "blue_noise.h"
#include "cubic_sampler.h"
#include "box_sampler.h"
#include "bspline3.h"
#include "mitchell_netravali.h"
#include "recfilter.h"
#if CUDA_SM < 20
#   include "cuPrintf.cu"
#   if __CUDA_ARCH__
#       define printf cuPrintf
#   endif
#endif

#define USE_LAUNCH_BOUNDS 1
const int BW_F1 = 32, // cuda block width
          BH_F1 = 8; 

const int BW_F2 = 32,
          BH_F2 = 8; 

#if USE_LAUNCH_BOUNDS
const int 
#if SAMPDIM == 8 && CUDA_SM >= 20
          NB_F1 = 2,  // number of blocks resident per SM
#else
          NB_F1 = 1,  // number of blocks resident per SM
#endif
          NB_F2 = 4;
#endif

__constant__ float2 blue_noise[SAMPDIM];

__constant__ float prefilter_data[SAMPDIM*KS*KS];

texture<float, 2, cudaReadModeElementType> t_aux_float;

struct texfetch_aux_float
{
    typedef float result_type;

    __device__ float operator()(float x, float y)
    {
        return tex2D(t_aux_float, x, y);
    }
};

__constant__ filter_operation filter_op;

// do the actual value processing according to what's in 'filter_op'
template <effect_type OP, class S>
__device__ typename S::result_type do_filter(const S &sampler, float2 pos)
{
    typename S::template rebind_sampler<texfetch_aux_float>::type sampler_aux_float;
    
    typedef typename S::result_type result_type;

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
        return threshold(sampler(pos),filter_op.minimum, filter_op.maximum);
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
    case EFFECT_HUE_SATURATION_LIGHTNESS:
        return hue_saturation_lightness(sampler(pos),filter_op.hue,
                                   filter_op.saturation,filter_op.lightness);
    case EFFECT_UNSHARP_MASK:
        return unsharp_mask(sampler(pos),sampler_aux_float(pos),
                            filter_op.amount,filter_op.threshold);
    case EFFECT_EMBOSS:
        {
            result_type i0 = sampler(pos),
                        i1 = sampler(pos+make_float2(-filter_op.offset,
                                                     filter_op.offset));
            return saturate(filter_op.amount*(i0-i1)+.5f);
        }
    case EFFECT_BILATERAL:
        {
            const float scale = 3*filter_op.sigma_s;
            // we're using sigma_r*3 to compesate something I don't understand
            const float inv_2sigma_r2 = 1.0f/(2*filter_op.sigma_r*filter_op.sigma_r),
                        inv_2sigma_s2 = 1.0f/(2*filter_op.sigma_s*filter_op.sigma_s);

            const result_type center = sampler(pos);
            result_type sum_weight = pixel_traits<result_type>::make_pixel(0), 
                        sum_color = pixel_traits<result_type>::make_pixel(0);

            const float space = 1.0f/8;

            for(int i=0; i<8; ++i)
            {
                for(int j=0; j<8; ++j)
                {
                    // d = [-0.5+1/16;0.5-1/16]
                    float2 d = make_float2(j+0.5f, i+0.5f)*space - 0.5f;

                    d *= scale;

                    float weight_s = expf(-(d.x*d.x + d.y*d.y)*inv_2sigma_s2);

                    result_type c = sampler(pos+d),
                                dc = center-c,
                                weight = expf(dc*dc*-inv_2sigma_r2)*weight_s;

                    sum_color += c*weight;
                    sum_weight += weight;
                }
            }
            return sum_color / sum_weight;
        }
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

struct filter_plan
{
    filter_plan() 
        : a_in(NULL)
        , a_aux_float(NULL)
        , prefilter_recfilter_plan(NULL)
        , gaussblur_plan(NULL)
    {
    }

    virtual ~filter_plan()
    {
        free(prefilter_recfilter_plan);

        if(a_in)
            cudaFreeArray(a_in);

        if(a_aux_float)
            cudaFreeArray(a_aux_float);

        free(gaussblur_plan);
    }

    int flags;
    cudaArray *a_in, *a_aux_float;
    dimage<float> img_aux_float_orig,
                  img_aux_float;

    filter_operation op;

    recfilter5_plan *prefilter_recfilter_plan;

    gaussian_blur_plan *gaussblur_plan;
};

template <int C>
struct filter_plan_C : filter_plan
{
    dimage<typename sum_traits<C>::type,KS*KS> temp_image;
};

template<int C>
void copy_to_array(cudaArray *out, dimage_ptr<const float,C> in);

void init_pre_filter(float (*prefilter)(float))
{
    std::vector<float2> blue_noise;
    std::vector<float> prefilter_data;
    blue_noise.reserve(SAMPDIM);
    prefilter_data.reserve(SAMPDIM*KS*KS);
    for(int i=0; i<SAMPDIM; ++i)
    {
        float2 n = make_float2(blue_noise_x[i], blue_noise_y[i]);

        blue_noise.push_back(n);
        for(int y=0; y<KS; ++y)
        {
            for(int x=0; x<KS; ++x)
            {
                prefilter_data.push_back(prefilter(x+n.x-1.5)*
                                         prefilter(y+n.y-1.5)/SAMPDIM);
            }
        }
    }

    copy_to_symbol("blue_noise",blue_noise);
    copy_to_symbol("prefilter_data",prefilter_data);
}

template<int C> 
filter_plan *
filter_create_plan(dimage_ptr<const float,C> img, const filter_operation &op,/*{{{*/
            int flags)
{
    assert(!img.empty());

    typedef filter_traits<C> cfg;
    typedef typename pixel_traits<float,C>::texel_type texel_type;
    typedef typename sum_traits<C>::type sum_type;

    filter_plan_C<C> *plan = new filter_plan_C<C>;

    plan->flags = flags;
    plan->op = op;

    int imgsize = img.width()*img.height();

    Vector<float,1+1> weights;

    // calculate cubic b-spline weights
    float a = 2.f-std::sqrt(3.0f);

    weights[0] = 1+a;
    weights[1] = a;

    base_timer *timer = NULL;

    // copy the input data to a texture
    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<texel_type>();

    cudaMallocArray(&plan->a_in, &ccd, img.width(),img.height());

    dimage<float,C> preproc_img;

    if(op.post_filter == FILTER_CARDINAL_BSPLINE3)
    {
        preproc_img.resize(img.width(), img.height());

        recfilter5_plan *postfilter_plan = 
            recfilter5_create_plan<1>(img.width(),img.height(),img.rowstride(),
                                      weights);
        try
        {
            if(flags & VERBOSE)
                timer = &timers.gpu_add("Convolve with bspline3^-1",
                                        img.width()*img.height(), "P");

            // convolve with a bpsline3^-1 to make a cardinal post-filter
            for(int i=0; i<C; ++i)
                recfilter5(postfilter_plan, preproc_img[i], img[i]);

            if(timer)
                timer->stop();

            copy_to_array(plan->a_in, dimage_ptr<const float,C>(&preproc_img));

            free(postfilter_plan);
        }
        catch(...)
        {
            free(postfilter_plan);
            throw;
        }
    }
    else
    {
        copy_to_array(plan->a_in, img);
        preproc_img = img;
    }

    if(op.pre_filter == FILTER_CARDINAL_BSPLINE3)
    {
        plan->prefilter_recfilter_plan = 
            recfilter5_create_plan<1>(img.width(),img.height(),img.rowstride(),
                                      weights);
    }

    cfg::tex().normalized = false;
    cfg::tex().filterMode = cudaFilterModeLinear;

    cfg::tex().addressMode[0] = cfg::tex().addressMode[1] = cudaAddressModeClamp;

    copy_to_symbol("filter_op",op);

    plan->temp_image.resize(img.width(), img.height());

    switch(op.pre_filter)
    {
    case FILTER_BSPLINE3:
    case FILTER_CARDINAL_BSPLINE3:
        init_pre_filter(&bspline3);
        break;
    case FILTER_MITCHELL_NETRAVALI:
        init_pre_filter(&mitchell_netravali);
        break;
    }

    switch(op.type)
    {
    case EFFECT_UNSHARP_MASK:
        ccd = cudaCreateChannelDesc<float>();
        cudaMallocArray(&plan->a_aux_float, &ccd, img.width(), img.height());
        check_cuda_error("cudaMallocArray");

        plan->img_aux_float_orig.resize(img.width(), img.height());
        luminance(&plan->img_aux_float_orig, &preproc_img);

        plan->gaussblur_plan 
            = gaussian_blur_create_plan(img.width(), img.height(), 
                                        img.rowstride(), op.sigma);

        plan->img_aux_float.resize(img.width(), img.height());

        t_aux_float.normalized = false;
        t_aux_float.filterMode = cudaFilterModeLinear;

        t_aux_float.addressMode[0] = t_aux_float.addressMode[1] = cudaAddressModeClamp;

        break;
    }

    return plan;
}/*}}}*/

void free(filter_plan *plan)/*{{{*/
{
    delete plan;
}/*}}}*/

template <class S, effect_type OP,int C>
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

    float *bspline3 = prefilter_data;

    S sampler;

    for(int s=0; s<SAMPDIM; ++s)
    {
        pixel_type value = do_filter<OP>(sampler, p+blue_noise[s]);
        value = srgb2lrgb(value);

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

template <class S, effect_type OP,int C>
__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW_F1*BH_F1, NB_F1)
#endif
void filter_kernel_box(dimage_ptr<float,C> out)/*{{{*/
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

    pixel_type sum = pixel_traits<float,C>::make_pixel(0);

    // top-left position of the kernel support
    float2 p = make_float2(x,y)+0.5f;

    S sampler;

    for(int s=0; s<SAMPDIM; ++s)
    {
        pixel_type value = do_filter<OP>(sampler, p+blue_noise[s]);
        sum += srgb2lrgb(value);
    }

    *out = sum/float(SAMPDIM);
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

template <class POST_FILTER, int C>
void filter(filter_plan *_plan, dimage_ptr<float,C> out, const filter_operation &op)/*{{{*/
{
    filter_plan_C<C> *plan = dynamic_cast<filter_plan_C<C> *>(_plan);
    assert(plan != NULL);

    if(plan->op.post_filter != op.post_filter)
        throw std::runtime_error("Postfilter changed, plan must be recreated");

    copy_to_symbol("filter_op",op);

    typedef filter_traits<C> cfg;
    assert(plan->temp_image.width() == out.width() &&
           plan->temp_image.height() == out.height());

    cudaBindTextureToArray(cfg::tex(), plan->a_in);

    dim3 bdim(BW_F1,BH_F1),
         gdim((out.width()+bdim.x-1)/bdim.x, (out.height()+bdim.y-1)/bdim.y);

    typedef filter_traits<C> cfg;

    base_timer *timer = NULL;

#define CASE(EFFECT) \
    case EFFECT:\
        if(plan->flags & VERBOSE)\
            timer = &timers.gpu_add("First pass",out.width()*out.height(),"P");\
        if(op.pre_filter == FILTER_BOX) \
            filter_kernel_box<POST_FILTER,EFFECT><<<gdim, bdim>>>(out); \
        else \
            filter_kernel1<POST_FILTER,EFFECT,C><<<gdim, bdim>>>(&plan->temp_image); \
        if(timer)\
            timer->stop();\
        break

    switch(op.type)
    {
    CASE(EFFECT_IDENTITY);
    CASE(EFFECT_POSTERIZE);
    CASE(EFFECT_SCALE);
    CASE(EFFECT_BIAS);
    CASE(EFFECT_ROOT);
    CASE(EFFECT_THRESHOLD);
    CASE(EFFECT_REPLACEMENT);
    CASE(EFFECT_GRADIENT_EDGE_DETECTION);
    CASE(EFFECT_LAPLACIAN);
    CASE(EFFECT_LAPLACE_EDGE_ENHANCEMENT);
    CASE(EFFECT_YAROSLAVSKY_BILATERAL);
    CASE(EFFECT_BRIGHTNESS_CONTRAST);
    CASE(EFFECT_HUE_SATURATION_LIGHTNESS);
    CASE(EFFECT_BILATERAL);
    CASE(EFFECT_EMBOSS);
    case EFFECT_UNSHARP_MASK:
        assert(plan->a_aux_float != NULL);

        if(plan->flags & VERBOSE)
            timer = &timers.gpu_add("Gaussian blur on Y channel",out.width()*out.height(),"P");

        update_plan(plan->gaussblur_plan, 
                    plan->img_aux_float_orig.width(),
                    plan->img_aux_float_orig.height(),
                    plan->img_aux_float_orig.rowstride(),
                    op.sigma);

        gaussian_blur(plan->gaussblur_plan, &plan->img_aux_float, 
                      &plan->img_aux_float_orig);
        if(timer)
            timer->stop();

        copy_to_array(plan->a_aux_float, dimage_ptr<const float>(&plan->img_aux_float));
        cudaBindTextureToArray(t_aux_float, plan->a_aux_float);

        if(plan->flags & VERBOSE)
            timer = &timers.gpu_add("First pass",out.width()*out.height(),"P");

        if(op.pre_filter == FILTER_BOX)
            filter_kernel_box<POST_FILTER,EFFECT_UNSHARP_MASK><<<gdim, bdim>>>(out);
        else
            filter_kernel1<POST_FILTER,EFFECT_UNSHARP_MASK,C><<<gdim, bdim>>>(&plan->temp_image);

        if(timer)
            timer->stop();

        cudaUnbindTexture(t_aux_float);
        break;
    default:
        assert(false);
    }
#undef CASE
                   
    if(op.pre_filter != FILTER_BOX)
    {
        if(plan->flags & VERBOSE)
            timer = &timers.gpu_add("Second pass",out.width()*out.height(),"P");

        dim3 bdim(BW_F2,BH_F2),
             gdim((out.width()+bdim.x-1)/bdim.x,(out.height()+bdim.y-1)/bdim.y);
        filter_kernel2<C><<<gdim, bdim>>>(out, &plan->temp_image);

        if(timer)
            timer->stop();
    }

    cudaUnbindTexture(cfg::tex());

    if(op.pre_filter == FILTER_CARDINAL_BSPLINE3)
    {
        if(plan->flags & VERBOSE)
            timer = &timers.gpu_add("Convolve with bspline3^-1",out.width()*out.height(),"P");

        // convolve with a bpsline3^-1 to make a cardinal pre-filter
        for(int i=0; i<C; ++i)
            recfilter5(plan->prefilter_recfilter_plan, out[i]);

        if(timer)
            timer->stop();
    }

    // maps back to gamma space
    lrgb2srgb(out, out);
}/*}}}*/

template <int C>
void filter(filter_plan *plan, dimage_ptr<float,C> out, const filter_operation &op)/*{{{*/
{
    typedef filter_traits<C> cfg;
    typedef typename cfg::texfetch_type texfetch;

    switch(op.post_filter)
    {
    case FILTER_BSPLINE3:
    case FILTER_CARDINAL_BSPLINE3:
        filter<cubic_sampler<bspline3_weights, texfetch>,C>(plan, out, op);
        break;
    case FILTER_MITCHELL_NETRAVALI:
        filter<cubic_sampler<mitchell_netravali_weights,texfetch>, C>(plan, out, op);
        break;
    case FILTER_BOX:
        filter<box_sampler<texfetch>, C>(plan, out, op);
        break;
    default:
        assert(false);
    }
}

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

    __device__ static float normalize_sum(float2 sum)
    {
        return sum.x / sum.y;
    }
};

template<>
void copy_to_array(cudaArray *out, dimage_ptr<const float> in)
{
    cudaMemcpy2DToArray(out, 0, 0, in, 
                        in.rowstride()*sizeof(float),
                        in.width()*sizeof(float), in.height(),
                        cudaMemcpyDeviceToDevice);
}

template 
void filter(filter_plan *, dimage_ptr<float,1> img, const filter_operation &op);

template 
filter_plan *
filter_create_plan(dimage_ptr<const float,1> img, const filter_operation &op, 
                   int flags);
/*}}}*/

//{{{ RGB filtering =========================================================

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

#if CUDA_SM >= 20
    static const int smem_size = 5;
#else
    static const int smem_size = 3;
#endif

    static int flags;

    static texture<float4,2,cudaReadModeElementType> &tex() 
        { return t_in_rgba; }

    __device__ static float3 normalize_sum(float4 sum)
    {
        return make_float3(sum) / sum.w;
    }
};

template <>
void copy_to_array(cudaArray *out, dimage_ptr<const float,3> img)
{
    dimage<float3> temp;
    temp.resize(img.width(), img.height());
    convert(&temp, img);

    cudaMemcpy2DToArray(out, 0, 0, temp, 
                        temp.rowstride()*sizeof(float4),
                        temp.width()*sizeof(float4), temp.height(),
                        cudaMemcpyDeviceToDevice);
}

template 
void filter(filter_plan *, dimage_ptr<float,3> img, const filter_operation &op);

template 
filter_plan *
filter_create_plan(dimage_ptr<const float,3> img, const filter_operation &op, 
                   int flags);
/*}}}*/
