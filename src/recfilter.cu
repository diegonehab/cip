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

#include "symbol.h"
#include "recfilter.h"

const int WS = 32;

#define ORDER 1
#include "alg5.cu"

#undef ORDER

#define ORDER 2
#include "alg5.cu"

struct recfilter5_plan
{
    recfilter5_plan() 
        // at least these should be initialized to make
        // upload_plan work when plan is empty
        : a_in(NULL)
        , width(0)
        , height(0)
        , border(0)
    {
    }

    virtual ~recfilter5_plan()
    {
        if(a_in != NULL)
            cudaFreeArray(a_in);
    }

    int width, height;
    int rowstride;
    float inv_width, inv_height;
    int m_size, n_size, last_m, last_n;
    BorderType border_type;
    int border;

    cudaArray *a_in;
};

template <int R>
struct recfilter5_plan_R : recfilter5_plan
{
    recfilter5_plan_R()
    {
        // this should be initialized for upload_plan
        for(int i=0; i<R; ++i)
            weights[i] = 0;
    }

    dvector<Matrix<float,R,WS> > d_pybar,
                                 d_ezhat,
                                 d_ptucheck,
                                 d_etvtilde;
    Vector<float, R+1> weights;

    Matrix<float,R,WS> AFP_T, ARE_T;
    Matrix<float,WS,WS> AFB_T, ARB_T;

    Matrix<float,R,R> AbF_T, AbR_T, AbF, AbR,
                      HARB_AFP_T, HARB_AFP;
    Matrix<float,R,WS> ARB_AFP_T, TAFB, HARB_AFB;
};

namespace
{
const recfilter5_plan *g_loaded_plan_in_gpu = NULL;

template <int R>
struct symbol_vars;

#define DEF_SYMBOL_VARS(N) \
template<> \
struct symbol_vars<N> \
{ \
    static const Vector<float,N+1> &weights; \
    static const Matrix<float,N,N> &AbF_T, &AbR_T, &HARB_AFP_T, \
                                   &AbF, &AbR, &HARB_AFP; \
    static const Matrix<float,N,32> &ARE_T, &HARB_AFB, \
                                    &TAFB, &ARB_AFP_T; \
};\
const Vector<float,N+1> &symbol_vars<N>::weights = ::c5_##N##_weights; \
const Matrix<float,N,N> &symbol_vars<N>::AbF_T = ::c5_##N##_AbF_T, \
                        &symbol_vars<N>::AbR_T = ::c5_##N##_AbR_T, \
                        &symbol_vars<N>::HARB_AFP_T = ::c5_##N##_HARB_AFP_T, \
                        &symbol_vars<N>::AbF = ::c5_##N##_AbF, \
                        &symbol_vars<N>::AbR = ::c5_##N##_AbR, \
                        &symbol_vars<N>::HARB_AFP = ::c5_##N##_HARB_AFP; \
const Matrix<float,N,32> &symbol_vars<N>::ARE_T = ::c5_##N##_ARE_T, \
                         &symbol_vars<N>::HARB_AFB = ::c5_##N##_HARB_AFB, \
                         &symbol_vars<N>::TAFB = ::c5_##N##_TAFB, \
                         &symbol_vars<N>::ARB_AFP_T = ::c5_##N##_ARB_AFP_T;

DEF_SYMBOL_VARS(1)
DEF_SYMBOL_VARS(2)

template<int R>
void load_plan(const recfilter5_plan_R<R> &plan)
{
    const recfilter5_plan_R<R> *gpu_plan
        = dynamic_cast<const recfilter5_plan_R<R> *>(g_loaded_plan_in_gpu);


    if(!gpu_plan || gpu_plan->weights != plan.weights)
    {
        typedef symbol_vars<R> sym;

        gpu::copy_to_symbol(sym::weights, plan.weights);

        gpu::copy_to_symbol(sym::AbF_T, plan.AbF_T);
        gpu::copy_to_symbol(sym::AbR_T, plan.AbR_T);
        gpu::copy_to_symbol(sym::HARB_AFP_T, plan.HARB_AFP_T);

        gpu::copy_to_symbol(sym::AbF, plan.AbF);
        gpu::copy_to_symbol(sym::AbR, plan.AbR);
        gpu::copy_to_symbol(sym::HARB_AFP, plan.HARB_AFP);

        gpu::copy_to_symbol(sym::ARE_T, plan.ARE_T);
        gpu::copy_to_symbol(sym::ARB_AFP_T, plan.ARB_AFP_T);
        gpu::copy_to_symbol(sym::TAFB, plan.TAFB);
        gpu::copy_to_symbol(sym::HARB_AFB, plan.HARB_AFB);
    }

    if(!g_loaded_plan_in_gpu || g_loaded_plan_in_gpu->border != plan.border)
        gpu::copy_to_symbol(c_border,plan.border);

    if(!g_loaded_plan_in_gpu || g_loaded_plan_in_gpu->rowstride!=plan.rowstride)
        gpu::copy_to_symbol(c_rowstride, plan.rowstride);

    if(!g_loaded_plan_in_gpu || g_loaded_plan_in_gpu->width != plan.width
       || g_loaded_plan_in_gpu->border != plan.border)
    {
        gpu::copy_to_symbol(c_width, plan.width); 
        gpu::copy_to_symbol(c_inv_width, plan.inv_width); 
        gpu::copy_to_symbol(c_m_size, plan.m_size); 
        gpu::copy_to_symbol(c_last_m, plan.last_m); 
    }

    if(!g_loaded_plan_in_gpu || g_loaded_plan_in_gpu->height != plan.height
       || g_loaded_plan_in_gpu->border != plan.border)
    {
        gpu::copy_to_symbol(c_inv_height, plan.inv_height);
        gpu::copy_to_symbol(c_height, plan.height);
        gpu::copy_to_symbol(c_n_size, plan.n_size);
        gpu::copy_to_symbol(c_last_n, plan.last_n);
    }

    if(!g_loaded_plan_in_gpu)
    {
        t_in.normalized = true;
        t_in.filterMode = cudaFilterModePoint;
    }

    if(!g_loaded_plan_in_gpu || g_loaded_plan_in_gpu->border_type != plan.border_type)
    {
        switch(plan.border_type)
        {
        case CLAMP_TO_ZERO:
            t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeBorder;
            break;
        case CLAMP_TO_EDGE:
            t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeClamp;
            break;
        case REPEAT:
            t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeWrap;
            break;
        case REFLECT:
            t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeMirror;
            break;
        }
    }

    g_loaded_plan_in_gpu = &plan;
}
} // local namespace


template <int R>
void recfilter5(recfilter5_plan_R<R> &plan,float *d_output,const float *d_input)
{
    load_plan(plan);

    cudaMemcpy2DToArray(plan.a_in, 0, 0, d_input, plan.rowstride*sizeof(float),
                        plan.width*sizeof(float), plan.height,
                      cudaMemcpyDeviceToDevice);

    cudaBindTextureToArray(t_in, plan.a_in);

    collect_carries<<< 
#if CUDA_SM >= 20
            dim3((plan.m_size+2-1)/2, plan.n_size), 
#else
            dim3(plan.m_size, plan.n_size), 
#endif
        dim3(WS, W1) >>>
        (&plan.d_pybar, &plan.d_ezhat, &plan.d_ptucheck, &plan.d_etvtilde);

    adjust_carries<<< dim3(1,plan.n_size), 
                     dim3(WS, std::min<int>(plan.m_size, W23)) >>>
        (&plan.d_pybar, &plan.d_ezhat, plan.m_size, plan.n_size );

    adjust_carries<<< dim3(plan.m_size,1), 
                     dim3(WS, std::min<int>(plan.n_size, W45)) >>>
        (&plan.d_ptucheck, &plan.d_etvtilde, &plan.d_pybar, &plan.d_ezhat, 
         plan.m_size, plan.n_size );

    write_result<<< 
#if CUDA_SM >= 20
            dim3((plan.m_size+2-1)/2,plan.n_size), 
#else
            dim3(plan.m_size,plan.n_size), 
#endif
                     dim3(WS, W6)>>>
        (d_output, &plan.d_pybar, &plan.d_ezhat, 
         &plan.d_ptucheck, &plan.d_etvtilde);

    cudaUnbindTexture(t_in);
}

void recfilter5(recfilter5_plan *plan, float *d_output, const float *d_input)
{
    assert(plan);

    if(recfilter5_plan_R<1> *plan_R = dynamic_cast<recfilter5_plan_R<1>*>(plan))
        recfilter5(*plan_R, d_output, d_input);
    else if(recfilter5_plan_R<2> *plan_R = dynamic_cast<recfilter5_plan_R<2>*>(plan))
        recfilter5(*plan_R, d_output, d_input);
    else
        throw std::runtime_error("Bad plan for recfilter5");
}

void recfilter5(recfilter5_plan *plan, float *d_inout)
{
    recfilter5(plan, d_inout, d_inout);
}

template <int R>
recfilter5_plan *
recfilter5_create_plan(int width, int height, int rowstride,
                       const Vector<float, R+1> &w, 
                       BorderType border_type, int border)
{
    recfilter5_plan_R<R> *plan = new recfilter5_plan_R<R>;
    try
    {
        update_plan<R>(plan, width, height, rowstride, w, border_type, border);

        load_plan(*plan);
    }
    catch(...)
    {
        delete plan;
        throw;
    }

    return plan;
}

template <int R>
void update_plan(recfilter5_plan *_plan, int width, int height, int rowstride,
                 const Vector<float, R+1> &w,
                 BorderType border_type, int border)
{
    assert(_plan);

    recfilter5_plan_R<R> *plan = dynamic_cast<recfilter5_plan_R<R> *>(_plan);
    if(plan == NULL)
        throw std::invalid_argument("Can't change recfilter's plan order");

    const int B = 32;


    int old_border = plan->border,
        old_width = plan->width,
        old_height = plan->height;

    if(old_width!=width || old_height!=height)
    {
        // let's do this first to at least have a passable strong
        // exception guarantee (this has more chance to blow up)

        cudaArray *a_in = NULL;

        cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
        cudaMallocArray(&a_in, &ccd, width, height);
        check_cuda_error("cudaMallocArray");

        try
        {
            if(plan->a_in)
            {
                cudaFreeArray(plan->a_in);
                check_cuda_error("cudaFreeArray");
                plan->a_in = NULL;
            }
        }
        catch(...)
        {
            cudaFreeArray(a_in);
            throw;
        }

        plan->a_in = a_in;
    }

    if(plan->weights != w)
    {
        Matrix<float,R,R> Ir = identity<float,R,R>();
        Matrix<float,B,R> Zbr = zeros<float,B,R>();
        Matrix<float,R,B> Zrb = zeros<float,R,B>();
        Matrix<float,B,B> Ib = identity<float,B,B>();

        // depends on weight
        plan->weights = w;
        plan->AFP_T = fwd(Ir, Zrb, w);
        plan->ARE_T = rev(Zrb, Ir, w);
        plan->AFB_T = fwd(Zbr, Ib, w);
        plan->ARB_T = rev(Ib, Zbr, w);

        plan->AbF_T = tail<R>(plan->AFP_T);
        plan->AbR_T = head<R>(plan->ARE_T);
        plan->AbF = transp(plan->AbF_T);
        plan->AbR = transp(plan->AbR_T);
        plan->HARB_AFP_T = plan->AFP_T*head<R>(plan->ARB_T);
        plan->HARB_AFP = transp(plan->HARB_AFP_T);
        plan->ARB_AFP_T = plan->AFP_T*plan->ARB_T;
        plan->TAFB = transp(tail<R>(plan->AFB_T));
        plan->HARB_AFB = transp(plan->AFB_T*head<R>(plan->ARB_T));
    }

    int bleft, bright, btop, bbottom;
    calc_borders(&bleft, &btop, &bright, &bbottom, width, height, border);

    // depends on width and border
    if(old_border != border || old_width != width)
    {
        plan->m_size = (width+bleft+bright+WS-1)/WS,
        plan->last_m = (bleft+width-1)/WS;
        plan->width = width;
        plan->inv_width = 1.f/width;
    }

    // depends on height and border
    if(old_border != border || old_height != height)
    {
        plan->n_size = (height+btop+bbottom+WS-1)/WS;
        plan->last_n = (btop+height-1)/WS;
        plan->height = height;
        plan->inv_height = 1.f/height;
    }

    // depends on width, height and border
    if(old_border!=border || old_width!=width || old_height!=height)
    {
        // TODO: provide strong exception guarantee of previous data
        // in case of any of these blowing up.

        plan->d_pybar.resize(plan->n_size*plan->m_size);
        plan->d_ezhat.resize(plan->n_size*plan->m_size);
        plan->d_ptucheck.resize(plan->n_size*plan->m_size);
        plan->d_etvtilde.resize(plan->n_size*plan->m_size);
    }


    // depends on rowstride
    plan->rowstride = rowstride;

    // depends on border
    plan->border_type = border_type;
    plan->border = border;
}


template
recfilter5_plan *
recfilter5_create_plan<1>(int width, int height, int rowstride,
                          const Vector<float, 1+1> &w, 
                          BorderType border_type, int border);

template
recfilter5_plan *
recfilter5_create_plan<2>(int width, int height, int rowstride,
                          const Vector<float, 2+1> &w, 
                          BorderType border_type, int border);

void free(recfilter5_plan *plan)
{
    if(g_loaded_plan_in_gpu == plan)
        g_loaded_plan_in_gpu = NULL;

    delete plan;
}


