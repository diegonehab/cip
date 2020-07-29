// Copyright 2010--2020 Andre Maximo, Rodolfo Schulz de Lima, Diego Nehab
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

//== INCLUDES =================================================================

#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <complex>
#include "config.h"

#include "symbol.h"
#include "dvector.h"
#include "util.h"

#include "recfilter.h"
#include "image.h"

#if CUDA_SM >= 20
#   define W1 8
#   define NB1 6

#   define W23 8
#   define NB23 6

#   define W45 7
#   define NB45 5

#   define W6 8
#   define NB6 7
#else
#   define W1 8
#   define NB1 4

#   define W23 8
#   define NB23 0

#   define W45 8
#   define NB45 0

#   define W6 8
#   define NB6 4
#endif

/*! @mainpage recursive-filtering

\section notes Notes

Naming conventions are: c_ constant; t_ texture; g_ global memory; s_
shared memory; d_ device pointer; a_ cuda-array; p_ template
parameter; f_ surface.

*/

#define PREFIX(x) PP_CAT3(c5_, ORDER,_##x)

#define c_weights PREFIX(weights)
#define c_AbF_T PREFIX(AbF_T)
#define c_AbR_T PREFIX(AbR_T)
#define c_HARB_AFP_T PREFIX(HARB_AFP_T)
#define c_AbF PREFIX(AbF)
#define c_AbR PREFIX(AbR)
#define c_HARB_AFP PREFIX(HARB_AFP)
#define c_ARE_T PREFIX(ARE_T)
#define c_HARB_AFB PREFIX(HARB_AFB)
#define c_TAFB PREFIX(TAFB)
#define c_ARB_AFP_T PREFIX(ARB_AFP_T)

#ifndef NON_ORDER_DEPENDENT_PARAMETERS_DEFINED
#define NON_ORDER_DEPENDENT_PARAMETERS_DEFINED 1
__constant__ int c_width, c_height, c_rowstride,
                 c_adj_width, c_adj_height,
                 c_m_size, // number of column-blocks,
                 c_n_size, // number of row-blocks,
                 c_last_m, c_last_n,
                 c_border;
__constant__ float c_inv_width, c_inv_height;
#endif

__constant__ Vector<float,ORDER+1> c_weights;

__constant__ Matrix<float,ORDER,ORDER> c_AbF_T, c_AbR_T, 
                                       c_HARB_AFP_T,
                               c_AbF, c_AbR, c_HARB_AFP;
__constant__ Matrix<float,ORDER,WS> c_ARE_T, c_HARB_AFB, 
                                    c_TAFB, c_ARB_AFP_T;


#ifndef TEXTURE_DEFINED
#define TEXTURE_DEFINED 1
texture<float, cudaTextureType2D, cudaReadModeElementType> t_in;
#endif

//=== IMPLEMENTATION ==========================================================

#ifndef AUX_FUNCS_DEFINED
#define AUX_FUNCS_DEFINED 1

template <int W, int U, int V>
__device__
void read_block(Matrix<float,U,V> &block, int m, int n, 
                float inv_width, float inv_height)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    float tu = ((m-c_border)*WS+tx+.5f)*inv_width,
          tv = ((n-c_border)*WS+ty+.5f)*inv_height;

    float (*bdata)[V] = (float (*)[V]) &block[ty][tx]
#if CUDA_SM >= 20
          ,(*bdata2)[V] = (float (*)[V])&block[ty+WS][tx]
#endif
          ;

#pragma unroll
    for(int i=0; i<WS-(WS%W); i+=W)
    {
        **bdata = tex2D(t_in, tu, tv);
        bdata += W;

#if CUDA_SM >= 20
        **bdata2 = tex2D(t_in, tu+WS*inv_width, tv);
        bdata2 += W;
#endif

        tv += W*inv_height;
    }

    if(ty < WS%W)
    {
        **bdata = tex2D(t_in, tu, tv);
#if CUDA_SM >= 20
        **bdata2 = tex2D(t_in, tu+WS*inv_width, tv);
#endif
    }
}

template <int W, int U, int V>
__device__
void write_block(float *out,
                 const Matrix<float,U,V> &block, 
                 int width, int height, int rowstride,
                 int m, int n, int last_m, int last_n)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    // current block intersects transp_out's area?
//    if(m >= c_border && m <= last_m && n >= c_border && n <= last_n)
    {
        int y = (n-c_border)*WS, 
            x = (m-c_border)*WS+tx;

        out += y*rowstride + x;

  //      if(y < height)
        {
            int maxy = min(height, y+WS);
            for(int i=0; y<maxy; ++y, ++i, out += width)
                *out = block[ty*WS+i][tx];

        }
    }
}

template <class T, int R>
__device__ 
Vector<T,R> mad(Matrix<T,R,WS> &r, const Vector<T,R> &a, 
                    const Matrix<T,R,R> &b)
{
#pragma unroll
    for(int j=0; j<R; ++j)
    {
        T acc = *r[j];
#pragma unroll
        for(int i=0; i<R; ++i)
            acc += a[i]*b[i][j];
        *r[j] = acc;
    }

    return r.col(0);
}

template <class T, int R>
__device__ 
Vector<T,R> mad(Matrix<T,R,WS> &r, const Matrix<T,R,R> &a,
                const Vector<T,R> &b)
{
#pragma unroll
    for(int i=0; i<R; ++i)
    {
        T acc = *r[i];
#pragma unroll
        for(int j=0; j<R; ++j)
            acc += a[i][j]*b[j];
        *r[i] = acc;
    }

    return r.col(0);
}

template <class T, int R>
__device__ 
void mad(Matrix<T,R,WS> &r, const Matrix<T,R,WS> &a, 
         const Matrix<T,R,R> &b)
{
#pragma unroll
    for(int j=0; j<R; ++j)
    {
        T acc = *r[j];
#pragma unroll
        for(int i=0; i<R; ++i)
            acc += *a[i]*b[i][j];
        *r[j] = acc;
    }
}

template <class T, int R>
__device__ 
void mad(Matrix<T,R,WS> &r, const Matrix<T,R,R> &a, const Matrix<T,R,WS> &b)
{
#pragma unroll
    for(int i=0; i<R; ++i)
    {
        T acc = *r[i];
#pragma unroll
        for(int j=0; j<R; ++j)
            acc += a[i][j]* *b[j];
        *r[i] = acc;
    }
}

template <class T, int R>
__device__ 
void mad(Matrix<T,R,WS> &r,  const Matrix<T,R,WS> &a, 
         const Matrix<T,R,WS> &b, const Matrix<T,R,WS> &c,
	    volatile T (*block_RD)[WS/2+WS+1])
{
    int tx = threadIdx.x, ty = threadIdx.y;

    Matrix<T,R,R> rint;

    for(int i=0; i<R; ++i)
    {
        for(int j=0; j<R; ++j)
        {
            block_RD[ty][tx] = a[i][tx] * *b[j];
            block_RD[ty][tx] += block_RD[ty][tx-1];
            block_RD[ty][tx] += block_RD[ty][tx-2];
            block_RD[ty][tx] += block_RD[ty][tx-4];
            block_RD[ty][tx] += block_RD[ty][tx-8];
            block_RD[ty][tx] += block_RD[ty][tx-16];
            rint[i][j] = block_RD[ty][WS-1];
        }
    }

    mad(r, rint, (const Matrix<T,R,WS> &)c[0][tx]);
}

#endif

/**
 *  @brief Algorithm 5 stage 1
 *
 *  This function computes the algorithm stage 5.1 following:
 *
 *  In parallel for all $m$ and $n$, compute and store each
 *  $P_{m,n}(\bar{Y})$, $E_{m,n}(\hat{Z})$, $P^\T_{m,n}(\check{U})$,
 *  and $E^\T_{m,n}(\tilde{V})$.
 *
 *  @param[in] g_in Input image
 *  @param[out] g_transp_ybar All P_{m,n}(\bar{Y})
 *  @param[out] g_transp_zhat All $E_{m,n}(\hat{Z})$
 *  @param[out] g_ucheck All $P^\T_{m,n}(\check{U})$
 *  @param[out] g_vtilde All $E^\T_{m,n}(\tilde{V})$
 */
__global__
#if NB1
__launch_bounds__(WS*W1, NB1)
#endif
void collect_carries(Matrix<float,ORDER,WS> *g_pybar, 
                     Matrix<float,ORDER,WS> *g_ezhat,
                     Matrix<float,ORDER,WS> *g_ptucheck, 
                     Matrix<float,ORDER,WS> *g_etvtilde)
{
    int tx = threadIdx.x, ty = threadIdx.y, 
#if CUDA_SM >= 20
        m = blockIdx.x*2, 
#else
        m = blockIdx.x, 
#endif
        n = blockIdx.y;

    // each cuda block will work on two horizontally adjacent WSxWS input data
    // blocks, so allocate enough shared memory for these.
#if CUDA_SM >= 20
    __shared__ Matrix<float,WS*2,WS+1> block;
#else
    __shared__ Matrix<float,WS,WS+1> block;
#endif

    // load data into shared memory
    read_block<W1>(block, m, n, c_inv_width, c_inv_height);

#if CUDA_SM >= 20
    m += ty;
    if(m >= c_m_size)
        return;
#endif

    __syncthreads();

#if CUDA_SM >= 20
    if(ty < 2)
#else
    if(ty == 0)
#endif
    {
        Matrix<float,ORDER,WS> 
            &pybar = (Matrix<float,ORDER,WS>&)g_pybar[n*c_m_size+m][0][tx],
            &ezhat = (Matrix<float,ORDER,WS>&)g_ezhat[n*c_m_size+m][0][tx],
            &ptucheck = (Matrix<float,ORDER,WS>&)g_ptucheck[n*c_m_size+m][0][tx],
            &etvtilde = (Matrix<float,ORDER,WS>&)g_etvtilde[n*c_m_size+m][0][tx];

        const float B0_1 = c_weights[0], B0_2 = B0_1*B0_1,
                    B0_3 = B0_2*B0_1, B0_4 = B0_2*B0_2;
        {
            float *bdata = block[tx+ty*WS];

            // calculate pybar, scan left -> right
            {
                Vector<float,ORDER> p = zeros<float,ORDER>();

                p[ORDER-1] = *bdata++;

#pragma unroll
                for(int j=1; j<WS; ++j, ++bdata)
                {
#if CUDA_SM >= 20 || ORDER>1
                    *bdata = fwd(p, *bdata, c_weights);
#else
                    *bdata = p[0] = rec_op(*bdata, p[0]*c_weights[1]);
#endif
                }

                if(m < c_m_size-1)
                    pybar.set_col(0, p*B0_1);
            }

            {
                --bdata;

                Vector<float,ORDER> e = zeros<float,ORDER>();

                e[0] = *bdata--;

#pragma unroll
                for(int j=WS-2; j>=0; --j, --bdata)
                {
#if CUDA_SM >= 20 || ORDER>1
                    *bdata = rev(*bdata, e, c_weights);
#else
                    *bdata = e[0] = rec_op(*bdata, e[0]*c_weights[1]);
#endif
                }

                if(m > 0)
                    ezhat.set_col(0, e*B0_2);
            }
        }

        {
            float (*bdata)[WS+1] = (float (*)[WS+1]) &block[ty*WS][tx];
            {
                Vector<float,ORDER> p = zeros<float,ORDER>();

                p[ORDER-1] = **bdata++;

#pragma unroll
                for(int i=1; i<WS; ++i, ++bdata)
                {
#if CUDA_SM >= 20 || ORDER>1
                    **bdata = fwd(p, **bdata, c_weights);
#else
                    **bdata = p[0] = rec_op(**bdata, p[0]*c_weights[1]);
#endif
                }

                if(n < c_n_size-1)
                    ptucheck.set_col(0, p*B0_3);
            }

            if(n > 0)
            {
                --bdata;

                Vector<float,ORDER> e = zeros<float,ORDER>();

                e[0] = **bdata--;

#pragma unroll
                for(int i=WS-2; i>=0; --i, --bdata)
                {
#if CUDA_SM >= 20 || ORDER>1
                    rev(**bdata, e, c_weights);
#else
                    e[0] = rec_op(**bdata, e[0]*c_weights[1]);
#endif
                }

                etvtilde.set_col(0, e*B0_4);
            }
        }
    }
}

/**
 *  @brief Algorithm 4 stage 2 and 3 (fusioned)
 *
 *  This function computes the algorithm stages 5.2 and 5.3 following:
 *
 *  In parallel for all $n$, sequentially for each $m$, compute and
 *  store the $P_{m,n}(Y)$ according to (37) and using the previously
 *  computed $P_{m-1,n}(\bar{Y})$.
 *
 *  with simple kernel fusioned (going thorough global memory):
 *
 *  In parallel for all $n$, sequentially for each $m$, compute and
 *  store $E_{m,n}(Z)$ according to (45) using the previously computed
 *  $P_{m-1,n}(Y)$ and $E_{m+1,n}(\hat{Z})$.
 *
 *  @param[in,out] g_transp_ybar All $P_{m,n}(\bar{Y})$
 *  @param[in,out] g_transp_zhat All $E_{m,n}(\hat{Z})$
 */
__global__
#if NB23
__launch_bounds__(WS*W23, NB23)
#endif
void adjust_carries(Matrix<float,ORDER,WS> *g_pybar, 
                    Matrix<float,ORDER,WS> *g_ezhat,
                    int m_size, int n_size)
{
    int tx = threadIdx.x, ty = threadIdx.y, n = blockIdx.y;

    __shared__ Matrix<float,ORDER,WS> block[W23];

    Matrix<float,ORDER,WS> &bdata = (Matrix<float,ORDER,WS> &)block[ty][0][tx];

    // P(ybar) -> P(y) processing --------------------------------------

    Matrix<float,ORDER,WS> *pybar = (Matrix<float,ORDER,WS> *)&g_pybar[n*m_size+ty][0][tx];

    Vector<float,ORDER> py = zeros<float,ORDER>(); // P(Y)

    int m = 0;
    if(blockDim.y == W23)
    {
        int mmax = m_size-(m_size%W23)-1;
        for(; m<mmax; m+=W23)
        {
            // read P(Y)
            bdata.set_col(0, pybar->col(0));

            __syncthreads();

            if(ty == 0)
            {
                Matrix<float,ORDER,WS> *bdata = (Matrix<float,ORDER,WS> *)&block[0][0][tx];
#pragma unroll
                for(int dm=0; dm<W23; ++dm, ++bdata)
                    py = mad(bdata[0], py, c_AbF_T);
            }

            __syncthreads();

            pybar->set_col(0,bdata.col(0));

            pybar += W23;
        }
    }

    // remaining column-blocks

    if(m < m_size-1)
    {
        if(m+ty < m_size-1)
            bdata.set_col(0, pybar->col(0));

        int remaining = m_size-1 - m;

        __syncthreads();

        if(ty == 0)
        {
            Matrix<float,ORDER,WS> *bdata = (Matrix<float,ORDER,WS> *)&block[0][0][tx];
#pragma unroll
            for(int dm=0; dm<remaining; ++dm, ++bdata)
                py = mad(bdata[0], py, c_AbF_T);
        }

        __syncthreads();

        if(m+ty < m_size-1)
            pybar->set_col(0,bdata.col(0));
    }


    // E(zhat) -> E(z) processing --------------------------------------

    m = m_size-1;

    Matrix<float,ORDER,WS> 
        *pm1y  = (Matrix<float,ORDER,WS> *)&g_pybar[n*m_size+m-ty-1][0][tx],
        *ezhat = (Matrix<float,ORDER,WS> *)&g_ezhat[n*m_size+m-ty][0][tx];


    // all pybars must be updated!
    __syncthreads();

    Vector<float,ORDER> ez = zeros<float,ORDER>();

    m = m_size-1;
    if(blockDim.y == W23)
    {
        int mmin = m_size%W23;
        for(; m>=mmin; m-=W23)
        {
            if(m > 0)
            {
                bdata.set_col(0, ezhat->col(0));

                if(m-ty > 0)
                    mad(bdata, *pm1y, c_HARB_AFP_T);

                __syncthreads();

                if(ty == 0)
                {
                    Matrix<float,ORDER,WS> *bdata 
                        = (Matrix<float,ORDER,WS> *)&block[0][0][tx];
#pragma unroll
                    for(int dm=0; dm<W23; ++dm, ++bdata)
                        ez = mad(bdata[0], ez, c_AbR_T);
                }

                __syncthreads();

                ezhat->set_col(0,bdata.col(0));
            }

            ezhat -= W23;
            pm1y -= W23;
        }
    }

    // remaining column-blocks

    if(m > 0)
    {
        int remaining = m+1;

        if(m-ty > 0)
        {
            bdata.set_col(0, ezhat->col(0));
            mad(bdata, *pm1y, c_HARB_AFP_T);
        }

        __syncthreads();

        if(ty == 0)
        {
            Matrix<float,ORDER,WS> *bdata = (Matrix<float,ORDER,WS> *)&block[0][0][tx];
#pragma unroll
            for(int dm=1; dm<remaining; ++dm, ++bdata)
                ez = mad(bdata[0], ez, c_AbR_T);
        }

        __syncthreads();

        if(m-ty > 0)
            ezhat->set_col(0,bdata.col(0));
    }
}

/**
 *  @brief Algorithm 5 stage 4 and 5 (fusioned)
 *
 *  This function computes the algorithm stages 5.2 and 5.3 following:
 *
 *  In parallel for all $n$, sequentially for each $m$, compute and
 *  store the $P_{m,n}(Y)$ according to (37) and using the previously
 *  computed $P_{m-1,n}(\bar{Y})$.
 *
 *  with simple kernel fusioned (going thorough global memory):
 *
 *  In parallel for all $n$, sequentially for each $m$, compute and
 *  store $E_{m,n}(Z)$ according to (45) using the previously computed
 *  $P_{m-1,n}(Y)$ and $E_{m+1,n}(\hat{Z})$.
 *
 *  @param[in,out] g_transp_ybar All $P_{m,n}(\bar{Y})$
 *  @param[in,out] g_transp_zhat All $E_{m,n}(\hat{Z})$
 */
__global__
#if NB45
__launch_bounds__(WS*W45, NB45)
#endif
void adjust_carries(Matrix<float,ORDER,WS> *g_ptucheck, 
                    Matrix<float,ORDER,WS> *g_etvtilde,
                    Matrix<float,ORDER,WS> *g_py, 
                    Matrix<float,ORDER,WS> *g_ez,

                    int m_size, int n_size)
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x;

    __shared__ Matrix<float,ORDER,WS> block[W45];

	volatile __shared__ float block_RD_raw[W45][WS/2+WS+1];
	volatile float (*block_RD)[WS/2+WS+1] = 
            (float (*)[WS/2+WS+1]) &block_RD_raw[0][WS/2];
    if(ty < W45)
        block_RD_raw[ty][tx] = 0;

    Matrix<float,ORDER,WS> &bdata = (Matrix<float,ORDER,WS> &)block[ty][0][tx];

    // Pt(ucheck) -> Pt(u) processing --------------------------------------

    Matrix<float,ORDER,WS> 
        *ptucheck = (Matrix<float,ORDER,WS> *)&g_ptucheck[ty*c_m_size+m][0][tx],
        *pm1y = (Matrix<float,ORDER,WS> *)&g_py[ty*c_m_size+m-1][0][tx],
        *em1z = (Matrix<float,ORDER,WS> *)&g_ez[ty*c_m_size+m+1][0][tx];

    Vector<float,ORDER> ptu = zeros<float,ORDER>(); // Pt(U)

    int n = 0;
    if(blockDim.y == W45)
    {
        int nmax = n_size-(n_size%W45)-1;
        for(; n<nmax; n+=W45)
        {
            // read Pt(U)
            bdata.set_col(0, ptucheck->col(0));

            if(m > 0)
                mad(bdata, c_TAFB, *pm1y, c_ARB_AFP_T, block_RD);

            if(m < c_m_size-1)
                mad(bdata, c_TAFB, *em1z, c_ARE_T, block_RD);

            __syncthreads();

            if(ty == 0)
            {
                Matrix<float,ORDER,WS> *bdata = (Matrix<float,ORDER,WS> *)&block[0][0][tx];
#pragma unroll
                for(int dn=0; dn<W45; ++dn, ++bdata)
                    ptu = mad(*bdata, c_AbF, ptu);
            }

            __syncthreads();

            ptucheck->set_col(0,bdata.col(0));

            ptucheck += W45*c_m_size;
            pm1y += W45*c_m_size;
            em1z += W45*c_m_size;
        }
    }

    // remaining column-blocks

    if(n < c_n_size-1)
    {
        if(n+ty < c_n_size-1)
        {
            bdata.set_col(0, ptucheck->col(0));

            if(m < c_m_size-1)
                mad(bdata, c_TAFB, *em1z, c_ARE_T, block_RD);

            if(m > 0)
                mad(bdata, c_TAFB, *pm1y, c_ARB_AFP_T, block_RD);
        }

        int remaining = n_size-1 - n;

        __syncthreads();

        if(ty == 0)
        {
            Matrix<float,ORDER,WS> *bdata = (Matrix<float,ORDER,WS> *)&block[0][0][tx];
#pragma unroll
            for(int dn=0; dn<remaining; ++dn, ++bdata)
                ptu = mad(bdata[0], c_AbF, ptu);
        }

        __syncthreads();

        if(n+ty < n_size-1)
            ptucheck->set_col(0,bdata.col(0));
    }

    // E(zhat) -> E(z) processing --------------------------------------

    n = n_size-1;

    Matrix<float,ORDER,WS>
        *etvtilde = (Matrix<float,ORDER,WS> *)&g_etvtilde[(n-ty)*c_m_size+m][0][tx],
        *ptn1u = (Matrix<float,ORDER,WS> *)&g_ptucheck[(n-ty-1)*c_m_size+m][0][tx];

    pm1y = (Matrix<float,ORDER,WS> *)&g_py[(n-ty)*c_m_size+m-1][0][tx];
    em1z = (Matrix<float,ORDER,WS> *)&g_ez[(n-ty)*c_m_size+m+1][0][tx];

    // all pybars must be updated!
    __syncthreads();

    Vector<float,ORDER> etv = zeros<float,ORDER>();

    if(blockDim.y == W45)
    {
        int nmin = n_size%W45;
        for(; n>=nmin; n-=W45)
        {
            if(n > 0)
            {
                bdata.set_col(0, etvtilde->col(0));

                if(m > 0)
                    mad(bdata, c_HARB_AFB, *pm1y, c_ARB_AFP_T, block_RD);

                if(m < c_m_size-1)
                    mad(bdata, c_HARB_AFB, *em1z, c_ARE_T, block_RD);

                if(n-ty > 0)
                    mad(bdata, *ptn1u, c_HARB_AFP_T);

                __syncthreads();

                if(ty == 0)
                {
                    Matrix<float,ORDER,WS> *bdata 
                        = (Matrix<float,ORDER,WS> *)&block[0][0][tx];
#pragma unroll
                    for(int dn=0; dn<W45; ++dn, ++bdata)
                        etv = mad(bdata[0], c_AbR, etv);
                }

                __syncthreads();

                etvtilde->set_col(0,bdata.col(0));
            }

            etvtilde -= W45*c_m_size;
            pm1y -= W45*c_m_size;
            em1z -= W45*c_m_size;
            ptn1u -= W45*c_m_size;
        }
    }

    // remaining column-blocks

    if(n > 0)
    {
        int remaining = n+1;

        if(n-ty > 0)
        {
            bdata.set_col(0, etvtilde->col(0));

            if(m > 0)
                mad(bdata, c_HARB_AFB, *pm1y, c_ARB_AFP_T, block_RD);

            if(m < c_m_size-1)
                mad(bdata, c_HARB_AFB, *em1z, c_ARE_T, block_RD);

            mad(bdata, *ptn1u, c_HARB_AFP_T);
        }

        __syncthreads();

        if(ty == 0)
        {
            Matrix<float,ORDER,WS> *bdata = (Matrix<float,ORDER,WS> *)&block[0][0][tx];
#pragma unroll
            for(int dn=1; dn<remaining; ++dn, ++bdata)
                etv = mad(bdata[0], c_AbR, etv);
        }

        __syncthreads();

        if(n-ty > 0)
            etvtilde->set_col(0,bdata.col(0));
    }
}

__global__
#if NB6
__launch_bounds__(WS*W6, NB6)
#endif
void write_result(float *g_out,
                  const Matrix<float,ORDER,WS> *g_py, 
                  const Matrix<float,ORDER,WS> *g_ez,
                  const Matrix<float,ORDER,WS> *g_ptu, 
                  const Matrix<float,ORDER,WS> *g_etv)
{
    int tx = threadIdx.x, ty = threadIdx.y, 
#if CUDA_SM >= 20
        m = blockIdx.x*2,
#else
        m = blockIdx.x,
#endif
        n = blockIdx.y;

    // each cuda block will work on two horizontally adjacent WSxWS input data
    // blocks, so allocate enough shared memory for these.
#if CUDA_SM >= 20
    __shared__ Matrix<float,WS*2,WS+1> block;
#else
    __shared__ Matrix<float,WS,WS+1> block;
#endif

    // load data into shared memory
    read_block<W6>(block, m, n, c_inv_width, c_inv_height);

#if CUDA_SM >= 20
    m += ty;
    if(m >= c_m_size)
        return;
#endif

    __syncthreads();

 #if CUDA_SM >= 20
    if(ty < 2)
#else
    if(ty == 0)
#endif
    {

        Matrix<float,ORDER,WS> 
            &py = (Matrix<float,ORDER,WS>&)  g_py[n*c_m_size+m-1][0][tx],
            &ez = (Matrix<float,ORDER,WS>&)  g_ez[n*c_m_size+m+1][0][tx],
            &ptu = (Matrix<float,ORDER,WS>&) g_ptu[(n-1)*c_m_size+m][0][tx],
            &etv = (Matrix<float,ORDER,WS>&) g_etv[(n+1)*c_m_size+m][0][tx];
        const float B0_2 = c_weights[0]*c_weights[0];

        {


            float *bdata = block[tx+ty*WS];

            // calculate pybar, scan left -> right
            Vector<float,ORDER> p = m==0 ? zeros<float,ORDER>()
                                     : py.col(0) / c_weights[0];

#pragma unroll
            for(int j=0; j<WS; ++j, ++bdata)
                *bdata = fwd(p, *bdata, c_weights);

            --bdata;

            Vector<float,ORDER> e = m==c_m_size-1 ? zeros<float,ORDER>()
                                              : ez.col(0);

#pragma unroll
            for(int j=WS-1; j>=0; --j, --bdata)
                *bdata = rev(*bdata*B0_2, e, c_weights);
        }
        {
            float (*bdata)[WS+1] = (float (*)[WS+1]) &block[ty*WS][tx];

            Vector<float,ORDER> p = n==0 ? zeros<float,ORDER>()
                                     : ptu.col(0) / c_weights[0];

#pragma unroll
            for(int i=0; i<WS; ++i, ++bdata)
                **bdata = fwd(p, **bdata, c_weights);

            --bdata;

            Vector<float,ORDER> e = n==c_n_size-1 ? zeros<float,ORDER>()
                                              : etv.col(0);

            // for some reason it's faster when this is here then inside the
            // next if block;
            int x = (m-c_border)*WS+tx;
            int y = (n-c_border+1)*WS-1;

            // current block intersects transp_out's area?
            if(m >= c_border && m <= c_last_m && n >= c_border && n <= c_last_n)
            {
                // image's end is in the middle of the block and we're outside
                // the image width?
                if(y >= c_height)
                {
                    // process data until we get into the image
                    int i;
#pragma unroll
                    for(i=y; i>=c_height; --i, --bdata)
                        rev(**bdata*B0_2, e, c_weights);

//                    bdata -= y-c_height+1;

                    // now we're inside the image, we must write to transp_out
                    float *out = g_out + (c_height-1)*c_rowstride + x;

                    int nmin = y-(WS-1);

#pragma unroll
                    for(;i>=nmin; --i, --bdata, out -= c_rowstride)
                    {
                        rev(**bdata*B0_2, e, c_weights);

                        if(x < c_width)
                            *out = e[0];
                    }
                }
                else
                {
                    float *out = g_out + y*c_rowstride + x;

#pragma unroll
                    for(int i=WS-1; i>=0; --i, --bdata, out -= c_rowstride)
                    {
                        rev(**bdata*B0_2, e, c_weights);

                        if(x < c_width)
                            *out = e[0];
                    }
                }
            }
        }
    }
}

