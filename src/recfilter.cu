#include "recfilter.h"

const int WS = 32;

#define ORDER 1
#include "alg5.cu"

#undef ORDER

#define ORDER 2
#include "alg5.cu"

template <int R>
struct recfilter5_plan_type
{
    dvector<Matrix<float,R,WS> > d_pybar,
                                 d_ezhat,
                                 d_ptucheck,
                                 d_etvtilde;
    int width, height;
    int rowstride;
    float inv_width, inv_height;
    int m_size, n_size;
    BorderType border_type;

    cudaArray *a_in;
};

recfilter5_plan_type<1> *plan_5_1 = NULL;
recfilter5_plan_type<2> *plan_5_2 = NULL;

/**
 *  @brief Recursive Filtering Algorithm 4 for filter order 2
 *
 *  This function computes the algorithm 4_2.
 *
 *  @param[in] h_img Input image
 *  @param[in] width Image width
 *  @param[in] height Image height
 *  @param[in] a0 Feedback coefficient
 *  @param[in] b1 Feedforward coefficient
 */
template <int R>
void recfilter5(float *d_output, const float *d_input,
                        recfilter5_plan_type<R> &plan)
{
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

void recfilter5(float *d_output, const float *d_input)
{
    if(plan_5_1 != NULL)
        recfilter5(d_output, d_input, *plan_5_1);
    else if(plan_5_2 != NULL)
        recfilter5(d_output, d_input, *plan_5_2);
    else
        throw std::runtime_error("Recursive filter plan not configured!");
}

void recfilter5(float *d_inout)
{
    recfilter5(d_inout, d_inout);
}

template <int R>
void recfilter5_setup(int width, int height, int rowstride,
                              const Vector<float, R+1> &w, 
                              BorderType border_type, int border,
                              recfilter5_plan_type<R> *&plan)
{
    const int B = 32;

    Matrix<float,R,R> Ir = identity<float,R,R>();
    Matrix<float,B,R> Zbr = zeros<float,B,R>();
    Matrix<float,R,B> Zrb = zeros<float,R,B>();
    Matrix<float,B,B> Ib = identity<float,B,B>();

    Matrix<float,R,B> AFP_T = fwd(Ir, Zrb, w),
                      ARE_T = rev(Zrb, Ir, w);
    Matrix<float,B,B> AFB_T = fwd(Zbr, Ib, w),
                      ARB_T = rev(Ib, Zbr, w);

    Matrix<float,R,R> AbF_T = tail<R>(AFP_T),
                      AbR_T = head<R>(ARE_T),
                      AbF = transp(AbF_T),
                      AbR = transp(AbR_T),
                      HARB_AFP_T = AFP_T*head<R>(ARB_T),
                      HARB_AFP = transp(HARB_AFP_T);
    Matrix<float,R,B> ARB_AFP_T = AFP_T*ARB_T,
                      TAFB = transp(tail<R>(AFB_T)),
                      HARB_AFB = transp(AFB_T*head<R>(ARB_T));


    int bleft, bright, btop, bbottom;
    calc_borders(&bleft, &btop, &bright, &bbottom, width, height, border);

    const int m_size = (width+bleft+bright+WS-1)/WS,
              n_size = (height+btop+bbottom+WS-1)/WS;

    int last_m = (bleft+width-1)/WS,
        last_n = (btop+height-1)/WS;
    float inv_width = 1.f/width, inv_height = 1.f/height;

    std::ostringstream ss;
    ss << "c5_" << R << "_";
    std::string prefix = ss.str();

    copy_to_symbol(prefix+"weights", w);

    copy_to_symbol(prefix+"AbF_T", AbF_T);
    copy_to_symbol(prefix+"AbR_T", AbR_T);
    copy_to_symbol(prefix+"HARB_AFP_T", HARB_AFP_T);

    copy_to_symbol(prefix+"AbF", AbF);
    copy_to_symbol(prefix+"AbR", AbR);
    copy_to_symbol(prefix+"HARB_AFP", HARB_AFP);

    copy_to_symbol(prefix+"ARE_T", ARE_T);
    copy_to_symbol(prefix+"ARB_AFP_T", ARB_AFP_T);
    copy_to_symbol(prefix+"TAFB", TAFB);
    copy_to_symbol(prefix+"HARB_AFB", HARB_AFB);

    copy_to_symbol(prefix+"border",border);
    copy_to_symbol(prefix+"inv_width", inv_width); 
    copy_to_symbol(prefix+"inv_height", inv_height);
    copy_to_symbol(prefix+"width", width); 
    copy_to_symbol(prefix+"height", height);
    copy_to_symbol(prefix+"m_size", m_size); 
    copy_to_symbol(prefix+"n_size", n_size);
    copy_to_symbol(prefix+"last_m", last_m); 
    copy_to_symbol(prefix+"last_n", last_n);
    copy_to_symbol(prefix+"rowstride", rowstride);

    t_in.normalized = true;
    t_in.filterMode = cudaFilterModePoint;

    switch(border_type)
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

    if(plan != NULL)
        recfilter5_free();

    plan = new recfilter5_plan_type<R>();

    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
    cudaMallocArray(&plan->a_in, &ccd, width, height);

    t_in.normalized = true;
    t_in.filterMode = cudaFilterModePoint;

    plan->rowstride = rowstride;

    plan->d_pybar.resize(n_size*m_size);
    plan->d_ezhat.resize(n_size*m_size);
    plan->d_ptucheck.resize(n_size*m_size);
    plan->d_etvtilde.resize(n_size*m_size);
    plan->border_type = border_type;

    plan->width = width;
    plan->height = height;
    plan->inv_width = inv_width;
    plan->inv_height = inv_height;
    plan->m_size = m_size;
    plan->n_size = n_size;
}

template <>
void recfilter5_setup<1>(int width, int height, int rowstride,
                              const Vector<float, 1+1> &w, 
                              BorderType border_type, int border)
{
    recfilter5_setup(width, height, rowstride, w, border_type, border,
                             plan_5_1);
}

template <>
void recfilter5_setup<2>(int width, int height, int rowstride,
                              const Vector<float, 2+1> &w, 
                              BorderType border_type, int border)
{
    recfilter5_setup(width, height, rowstride, w, border_type, border,
                             plan_5_2);
}

void recfilter5_free()
{
    if(plan_5_1)
    {
        cudaFreeArray(plan_5_1->a_in);
        delete plan_5_1;
        plan_5_1 = NULL;
    }
    if(plan_5_2)
    {
        cudaFreeArray(plan_5_2->a_in);
        delete plan_5_2;
        plan_5_2 = NULL;
    }
}


