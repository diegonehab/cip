/**
 *  @file alg4-filtering.cu
 *  @brief GPU-Efficient Recursive Filtering Kernels - algorithm 5
 *  @author Diego Nehab
 *  @author Andre Maximo
 *  @author Rodolfo Lima
 *  @date October, 2010
 *  @date September, 2011
 *  @copyright The MIT License
 */

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

#include "recfilter.h"
#include "image.h"


const int WS = 32;

#if USE_SM20
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

__constant__ int c5_width, c5_height, c5_rowstride,
                 c5_adj_width, c5_adj_height,
                 c5_m_size, // number of column-blocks,
                 c5_n_size, // number of row-blocks,
                 c5_last_m, c5_last_n,
                 c_border;
__constant__ float c5_inv_with, c5_inv_height;

__constant__ Vector<float,ORDER+1> c5_weights;

__constant__ Matrix<float,ORDER,ORDER> c5_AbF_T, c5_AbR_T, c5_HARB_AFP_T,
                               c5_AbF, c5_AbR, c5_HARB_AFP;
__constant__ Matrix<float,ORDER,WS> c5_ARE_T, c5_HARB_AFB, c5_TAFB, c5_ARB_AFP_T;

texture<float, cudaTextureType2D, cudaReadModeElementType> t_in;

//=== IMPLEMENTATION ==========================================================

template <int W, int U, int V>
__device__
void read_block(Matrix<float,U,V> &block, int m, int n, 
                float inv_width, float inv_height)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    float tu = ((m-c_border)*WS+tx+.5f)*inv_width,
          tv = ((n-c_border)*WS+ty+.5f)*inv_height;

    float (*bdata)[V] = (float (*)[V]) &block[ty][tx]
#if USE_SM20
          ,(*bdata2)[V] = (float (*)[V])&block[ty+WS][tx]
#endif
          ;

#pragma unroll
    for(int i=0; i<WS-(WS%W); i+=W)
    {
        **bdata = tex2D(t_in, tu, tv);
        bdata += W;

#if USE_SM20
        **bdata2 = tex2D(t_in, tu+WS*inv_width, tv);
        bdata2 += W;
#endif

        tv += W*inv_height;
    }

    if(ty < WS%W)
    {
        **bdata = tex2D(t_in, tu, tv);
#if USE_SM20
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
template <int R>
__global__
#if NB1
__launch_bounds__(WS*W1, NB1)
#endif
void collect_carries(Matrix<float,R,WS> *g_pybar, 
                     Matrix<float,R,WS> *g_ezhat,
                     Matrix<float,R,WS> *g_ptucheck, 
                     Matrix<float,R,WS> *g_etvtilde)
{
    int tx = threadIdx.x, ty = threadIdx.y, 
#if USE_SM20
        m = blockIdx.x*2, 
#else
        m = blockIdx.x, 
#endif
        n = blockIdx.y;

    // each cuda block will work on two horizontally adjacent WSxWS input data
    // blocks, so allocate enough shared memory for these.
#if USE_SM20
    __shared__ Matrix<float,WS*2,WS+1> block;
#else
    __shared__ Matrix<float,WS,WS+1> block;
#endif

    // load data into shared memory
    read_block<W1>(block, m, n, c5_inv_with, c5_inv_height);

#if USE_SM20
    m += ty;
    if(m >= c5_m_size)
        return;
#endif

    __syncthreads();

#if USE_SM20
    if(ty < 2)
#else
    if(ty == 0)
#endif
    {

        Matrix<float,R,WS> 
            &pybar = (Matrix<float,R,WS>&)g_pybar[n*c5_m_size+m][0][tx],
            &ezhat = (Matrix<float,R,WS>&)g_ezhat[n*c5_m_size+m][0][tx],
            &ptucheck = (Matrix<float,R,WS>&)g_ptucheck[n*c5_m_size+m][0][tx],
            &etvtilde = (Matrix<float,R,WS>&)g_etvtilde[n*c5_m_size+m][0][tx];

        const float B0_1 = c5_weights[0], B0_2 = B0_1*B0_1,
                    B0_3 = B0_2*B0_1, B0_4 = B0_2*B0_2;
        {
            float *bdata = block[tx+ty*WS];

            // calculate pybar, scan left -> right
            {
                Vector<float,R> p = zeros<float,R>();

                p[R-1] = *bdata++;

#pragma unroll
                for(int j=1; j<WS; ++j, ++bdata)
                {
#if USE_SM20 || ORDER>1
                    *bdata = fwd(p, *bdata, c5_weights);
#else
                    *bdata = p[0] = rec_op(*bdata, p[0]*c5_weights[1]);
#endif
                }

                if(m < c5_m_size-1)
                    pybar.set_col(0, p*B0_1);
            }

            {
                --bdata;

                Vector<float,R> e = zeros<float,R>();

                e[0] = *bdata--;

                for(int j=WS-2; j>=0; --j, --bdata)
                {
#if USE_SM20 || ORDER>1
                    *bdata = rev(*bdata, e, c5_weights);
#else
                    *bdata = e[0] = rec_op(*bdata, e[0]*c5_weights[1]);
#endif
                }

                if(m > 0)
                    ezhat.set_col(0, e*B0_2);
            }
        }

        {
            float (*bdata)[WS+1] = (float (*)[WS+1]) &block[ty*WS][tx];
            {
                Vector<float,R> p = zeros<float,R>();

                p[R-1] = **bdata++;

#pragma unroll
                for(int i=1; i<WS; ++i, ++bdata)
                {
#if USE_SM20 || ORDER>1
                    **bdata = fwd(p, **bdata, c5_weights);
#else
                    **bdata = p[0] = rec_op(**bdata, p[0]*c5_weights[1]);
#endif
                }

                if(n < c5_n_size-1)
                    ptucheck.set_col(0, p*B0_3);
            }

            if(n > 0)
            {
                --bdata;

                Vector<float,R> e = zeros<float,R>();

                e[0] = **bdata--;

#pragma unroll
                for(int i=WS-2; i>=0; --i, --bdata)
                {
#if USE_SM20 || ORDER>1
                    rev(**bdata, e, c5_weights);
#else
                    e[0] = rec_op(**bdata, e[0]*c5_weights[1]);
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
template <int R>
__global__
#if NB23
__launch_bounds__(WS*W23, NB23)
#endif
void adjust_carries(Matrix<float,R,WS> *g_pybar, 
                    Matrix<float,R,WS> *g_ezhat,
                    int m_size, int n_size)
{
    int tx = threadIdx.x, ty = threadIdx.y, n = blockIdx.y;

    __shared__ Matrix<float,R,WS> block[W23];

    Matrix<float,R,WS> &bdata = (Matrix<float,R,WS> &)block[ty][0][tx];

    // P(ybar) -> P(y) processing --------------------------------------

    Matrix<float,R,WS> *pybar = (Matrix<float,R,WS> *)&g_pybar[n*m_size+ty][0][tx];

    Vector<float,R> py = zeros<float,R>(); // P(Y)

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
                Matrix<float,R,WS> *bdata = (Matrix<float,R,WS> *)&block[0][0][tx];
#pragma unroll
                for(int dm=0; dm<W23; ++dm, ++bdata)
                    py = mad(bdata[0], py, c5_AbF_T);
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
            Matrix<float,R,WS> *bdata = (Matrix<float,R,WS> *)&block[0][0][tx];
#pragma unroll
            for(int dm=0; dm<remaining; ++dm, ++bdata)
                py = mad(bdata[0], py, c5_AbF_T);
        }

        __syncthreads();

        if(m+ty < m_size-1)
            pybar->set_col(0,bdata.col(0));
    }


    // E(zhat) -> E(z) processing --------------------------------------

    m = m_size-1;

    Matrix<float,R,WS> 
        *pm1y  = (Matrix<float,R,WS> *)&g_pybar[n*m_size+m-ty-1][0][tx],
        *ezhat = (Matrix<float,R,WS> *)&g_ezhat[n*m_size+m-ty][0][tx];


    // all pybars must be updated!
    __syncthreads();

    Vector<float,R> ez = zeros<float,R>();

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
                    mad(bdata, *pm1y, c5_HARB_AFP_T);

                __syncthreads();

                if(ty == 0)
                {
                    Matrix<float,R,WS> *bdata 
                        = (Matrix<float,R,WS> *)&block[0][0][tx];
#pragma unroll
                    for(int dm=0; dm<W23; ++dm, ++bdata)
                        ez = mad(bdata[0], ez, c5_AbR_T);
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
            mad(bdata, *pm1y, c5_HARB_AFP_T);
        }

        __syncthreads();

        if(ty == 0)
        {
            Matrix<float,R,WS> *bdata = (Matrix<float,R,WS> *)&block[0][0][tx];
#pragma unroll
            for(int dm=1; dm<remaining; ++dm, ++bdata)
                ez = mad(bdata[0], ez, c5_AbR_T);
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
template <int R>
__global__
#if NB45
__launch_bounds__(WS*W45, NB45)
#endif
void adjust_carries(Matrix<float,R,WS> *g_ptucheck, 
                    Matrix<float,R,WS> *g_etvtilde,
                    Matrix<float,R,WS> *g_py, 
                    Matrix<float,R,WS> *g_ez,

                    int m_size, int n_size)
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x;

    __shared__ Matrix<float,R,WS> block[W45];

	volatile __shared__ float block_RD_raw[W45][WS/2+WS+1];
	volatile float (*block_RD)[WS/2+WS+1] = 
            (float (*)[WS/2+WS+1]) &block_RD_raw[0][WS/2];
    if(ty < W45)
        block_RD_raw[ty][tx] = 0;

    Matrix<float,R,WS> &bdata = (Matrix<float,R,WS> &)block[ty][0][tx];

    // Pt(ucheck) -> Pt(u) processing --------------------------------------

    Matrix<float,R,WS> 
        *ptucheck = (Matrix<float,R,WS> *)&g_ptucheck[ty*c5_m_size+m][0][tx],
        *pm1y = (Matrix<float,R,WS> *)&g_py[ty*c5_m_size+m-1][0][tx],
        *em1z = (Matrix<float,R,WS> *)&g_ez[ty*c5_m_size+m+1][0][tx];

    Vector<float,R> ptu = zeros<float,R>(); // Pt(U)

    int n = 0;
    if(blockDim.y == W45)
    {
        int nmax = n_size-(n_size%W45)-1;
        for(; n<nmax; n+=W45)
        {
            // read Pt(U)
            bdata.set_col(0, ptucheck->col(0));

            if(m > 0)
                mad(bdata, c5_TAFB, *pm1y, c5_ARB_AFP_T, block_RD);

            if(m < c5_m_size-1)
                mad(bdata, c5_TAFB, *em1z, c5_ARE_T, block_RD);

            __syncthreads();

            if(ty == 0)
            {
                Matrix<float,R,WS> *bdata = (Matrix<float,R,WS> *)&block[0][0][tx];
#pragma unroll
                for(int dn=0; dn<W45; ++dn, ++bdata)
                    ptu = mad(*bdata, c5_AbF, ptu);
            }

            __syncthreads();

            ptucheck->set_col(0,bdata.col(0));

            ptucheck += W45*c5_m_size;
            pm1y += W45*c5_m_size;
            em1z += W45*c5_m_size;
        }
    }

    // remaining column-blocks

    if(n < c5_n_size-1)
    {
        if(n+ty < c5_n_size-1)
        {
            bdata.set_col(0, ptucheck->col(0));

            if(m < c5_m_size-1)
                mad(bdata, c5_TAFB, *em1z, c5_ARE_T, block_RD);

            if(m > 0)
                mad(bdata, c5_TAFB, *pm1y, c5_ARB_AFP_T, block_RD);
        }

        int remaining = n_size-1 - n;

        __syncthreads();

        if(ty == 0)
        {
            Matrix<float,R,WS> *bdata = (Matrix<float,R,WS> *)&block[0][0][tx];
#pragma unroll
            for(int dn=0; dn<remaining; ++dn, ++bdata)
                ptu = mad(bdata[0], c5_AbF, ptu);
        }

        __syncthreads();

        if(n+ty < n_size-1)
            ptucheck->set_col(0,bdata.col(0));
    }

    // E(zhat) -> E(z) processing --------------------------------------

    n = n_size-1;

    Matrix<float,R,WS>
        *etvtilde = (Matrix<float,R,WS> *)&g_etvtilde[(n-ty)*c5_m_size+m][0][tx],
        *ptn1u = (Matrix<float,R,WS> *)&g_ptucheck[(n-ty-1)*c5_m_size+m][0][tx];

    pm1y = (Matrix<float,R,WS> *)&g_py[(n-ty)*c5_m_size+m-1][0][tx];
    em1z = (Matrix<float,R,WS> *)&g_ez[(n-ty)*c5_m_size+m+1][0][tx];

    // all pybars must be updated!
    __syncthreads();

    Vector<float,R> etv = zeros<float,R>();

    if(blockDim.y == W45)
    {
        int nmin = n_size%W45;
        for(; n>=nmin; n-=W45)
        {
            if(n > 0)
            {
                bdata.set_col(0, etvtilde->col(0));

                if(m > 0)
                    mad(bdata, c5_HARB_AFB, *pm1y, c5_ARB_AFP_T, block_RD);

                if(m < c5_m_size-1)
                    mad(bdata, c5_HARB_AFB, *em1z, c5_ARE_T, block_RD);

                if(n-ty > 0)
                    mad(bdata, *ptn1u, c5_HARB_AFP_T);

                __syncthreads();

                if(ty == 0)
                {
                    Matrix<float,R,WS> *bdata 
                        = (Matrix<float,R,WS> *)&block[0][0][tx];
#pragma unroll
                    for(int dn=0; dn<W45; ++dn, ++bdata)
                        etv = mad(bdata[0], c5_AbR, etv);
                }

                __syncthreads();

                etvtilde->set_col(0,bdata.col(0));
            }

            etvtilde -= W45*c5_m_size;
            pm1y -= W45*c5_m_size;
            em1z -= W45*c5_m_size;
            ptn1u -= W45*c5_m_size;
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
                mad(bdata, c5_HARB_AFB, *pm1y, c5_ARB_AFP_T, block_RD);

            if(m < c5_m_size-1)
                mad(bdata, c5_HARB_AFB, *em1z, c5_ARE_T, block_RD);

            mad(bdata, *ptn1u, c5_HARB_AFP_T);
        }

        __syncthreads();

        if(ty == 0)
        {
            Matrix<float,R,WS> *bdata = (Matrix<float,R,WS> *)&block[0][0][tx];
#pragma unroll
            for(int dn=1; dn<remaining; ++dn, ++bdata)
                etv = mad(bdata[0], c5_AbR, etv);
        }

        __syncthreads();

        if(n-ty > 0)
            etvtilde->set_col(0,bdata.col(0));
    }
}

template <int R>
__global__
#if NB6
__launch_bounds__(WS*W6, NB6)
#endif
void write_result(float *g_out,
                  const Matrix<float,R,WS> *g_py, 
                  const Matrix<float,R,WS> *g_ez,
                  const Matrix<float,R,WS> *g_ptu, 
                  const Matrix<float,R,WS> *g_etv)
{
    int tx = threadIdx.x, ty = threadIdx.y, 
#if USE_SM20
        m = blockIdx.x*2,
#else
        m = blockIdx.x,
#endif
        n = blockIdx.y;

    // each cuda block will work on two horizontally adjacent WSxWS input data
    // blocks, so allocate enough shared memory for these.
#if USE_SM20
    __shared__ Matrix<float,WS*2,WS+1> block;
#else
    __shared__ Matrix<float,WS,WS+1> block;
#endif

    // load data into shared memory
    read_block<W6>(block, m, n, c5_inv_with, c5_inv_height);

#if USE_SM20
    m += ty;
    if(m >= c5_m_size)
        return;
#endif

    __syncthreads();

 #if USE_SM20
    if(ty < 2)
#else
    if(ty == 0)
#endif
    {

        Matrix<float,R,WS> 
            &py = (Matrix<float,R,WS>&)  g_py[n*c5_m_size+m-1][0][tx],
            &ez = (Matrix<float,R,WS>&)  g_ez[n*c5_m_size+m+1][0][tx],
            &ptu = (Matrix<float,R,WS>&) g_ptu[(n-1)*c5_m_size+m][0][tx],
            &etv = (Matrix<float,R,WS>&) g_etv[(n+1)*c5_m_size+m][0][tx];
        const float B0_2 = c5_weights[0]*c5_weights[0];

        {


            float *bdata = block[tx+ty*WS];

            // calculate pybar, scan left -> right
            Vector<float,R> p = m==0 ? zeros<float,R>()
                                     : py.col(0) / c5_weights[0];

#pragma unroll
            for(int j=0; j<WS; ++j, ++bdata)
                *bdata = fwd(p, *bdata, c5_weights);

                --bdata;

            Vector<float,R> e = m==c5_m_size-1 ? zeros<float,R>()
                                              : ez.col(0);

#pragma unroll
            for(int j=WS-1; j>=0; --j, --bdata)
                *bdata = rev(*bdata*B0_2, e, c5_weights);
        }
        {
            float (*bdata)[WS+1] = (float (*)[WS+1]) &block[ty*WS][tx];

            Vector<float,R> p = n==0 ? zeros<float,R>()
                                     : ptu.col(0) / c5_weights[0];

#pragma unroll
            for(int i=0; i<WS; ++i, ++bdata)
                **bdata = fwd(p, **bdata, c5_weights);

            --bdata;

            Vector<float,R> e = n==c5_n_size-1 ? zeros<float,R>()
                                              : etv.col(0);

            // for some reason it's faster when this is here then inside the
            // next if block;
            int x = (m-c_border)*WS+tx;
            int y = (n-c_border+1)*WS-1;

            // current block intersects transp_out's area?
            if(m >= c_border && m <= c5_last_m && n >= c_border && n <= c5_last_n)
            {
                // image's end is in the middle of the block and we're outside
                // the image width?
                if(y >= c5_height)
                {
                    // process data until we get into the image
                    int i;
#pragma unroll
                    for(i=y; i>=c5_height; --i, --bdata)
                        rev(**bdata*B0_2, e, c5_weights);

//                    bdata -= y-c5_height+1;

                    // now we're inside the image, we must write to transp_out
                    float *out = g_out + (c5_height-1)*c5_rowstride + x;

                    int nmin = y-(WS-1);

#pragma unroll
                    for(;i>=nmin; --i, --bdata, out -= c5_rowstride)
                    {
                        rev(**bdata*B0_2, e, c5_weights);

                        if(x < c5_width)
                            *out = e[0];
                    }
                }
                else
                {
                    float *out = g_out + y*c5_rowstride + x;

#pragma unroll
                    for(int i=WS-1; i>=0; --i, --bdata, out -= c5_rowstride)
                    {
                        rev(**bdata*B0_2, e, c5_weights);

                        if(x < c5_width)
                            *out = e[0];
                    }
                }
            }
        }
    }
}

struct recfilter5_plan_type
{
    dvector<Matrix<float,ORDER,WS> > d_pybar,
                                 d_ezhat,
                                 d_ptucheck,
                                 d_etvtilde;
    int width, height;
    int rowstride;
    float inv_width, inv_height;
    int m_size, n_size;
    BorderType border_type;

    cudaArray *a_in;
} *plan = NULL;

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
__host__
void recursive_filter_5(dvector<float> &d_output, const dvector<float> &d_input)
{
    assert(d_input.size() >= plan->rowstride*plan->height);
    if(d_input.size() < plan->rowstride*plan->height)
        throw std::runtime_error("Invalid image size");

    d_output.resize(d_input.size());

    cudaMemcpy2DToArray(plan->a_in, 0, 0, d_input, plan->rowstride*sizeof(float),
                        plan->width*sizeof(float), plan->height,
                      cudaMemcpyDeviceToDevice);

    cudaBindTextureToArray(t_in, plan->a_in);

    collect_carries<<< 
#if USE_SM20
            dim3((plan->m_size+2-1)/2, plan->n_size), 
#else
            dim3(plan->m_size, plan->n_size), 
#endif
        dim3(WS, W1) >>>
        (&plan->d_pybar, &plan->d_ezhat, &plan->d_ptucheck, &plan->d_etvtilde);

    adjust_carries<<< dim3(1,plan->n_size), 
                     dim3(WS, std::min<int>(plan->m_size, W23)) >>>
        (&plan->d_pybar, &plan->d_ezhat, plan->m_size, plan->n_size );

    adjust_carries<<< dim3(plan->m_size,1), 
                     dim3(WS, std::min<int>(plan->n_size, W45)) >>>
        (&plan->d_ptucheck, &plan->d_etvtilde, &plan->d_pybar, &plan->d_ezhat, 
         plan->m_size, plan->n_size );

    write_result<<< 
#if USE_SM20
            dim3((plan->m_size+2-1)/2,plan->n_size), 
#else
            dim3(plan->m_size,plan->n_size), 
#endif
                     dim3(WS, W6)>>>
        (&d_output, &plan->d_pybar, &plan->d_ezhat, 
         &plan->d_ptucheck, &plan->d_etvtilde);

    cudaUnbindTexture(t_in);

}

void recursive_filter_5(dvector<float> &d_inout)
{
    recursive_filter_5(d_inout, d_inout);
}

void recursive_filter_5_setup(int width, int height, int rowstride,
                              const Vector<float, ORDER+1> &w, 
                              BorderType border_type, int border)
{
    const int R = ORDER;
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

    copy_to_symbol("c5_weights", w);

    copy_to_symbol("c5_AbF_T", AbF_T);
    copy_to_symbol("c5_AbR_T", AbR_T);
    copy_to_symbol("c5_HARB_AFP_T", HARB_AFP_T);

    copy_to_symbol("c5_AbF", AbF);
    copy_to_symbol("c5_AbR", AbR);
    copy_to_symbol("c5_HARB_AFP", HARB_AFP);

    copy_to_symbol("c5_ARE_T", ARE_T);
    copy_to_symbol("c5_ARB_AFP_T", ARB_AFP_T);
    copy_to_symbol("c5_TAFB", TAFB);
    copy_to_symbol("c5_HARB_AFB", HARB_AFB);

    copy_to_symbol("c_border",border);
    copy_to_symbol("c5_inv_with", inv_width); 
    copy_to_symbol("c5_inv_height", inv_height);
    copy_to_symbol("c5_width", width); 
    copy_to_symbol("c5_height", height);
    copy_to_symbol("c5_m_size", m_size); 
    copy_to_symbol("c5_n_size", n_size);
    copy_to_symbol("c5_last_m", last_m); 
    copy_to_symbol("c5_last_n", last_n);
    copy_to_symbol("c5_rowstride", rowstride);

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
        recursive_filter_5_free();

    plan = new recfilter5_plan_type();


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

void recursive_filter_5_free()
{
    if(plan)
    {
        cudaFreeArray(plan->a_in);
        delete plan;
        plan = NULL;
    }
}


