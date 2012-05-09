#ifndef NLFILTER_BSPLINE3_SAMPLER_H
#define NLFILTER_BSPLINE3_SAMPLER_H

template <class T>
__device__ void bspline3_weights(T alpha, T &w0, T &w1, T &w2, T &w3,
                                 int k=0)
{
    T one_alpha = 1.0f - alpha,
      alpha2 = alpha * alpha,
      one_alpha2 = one_alpha*one_alpha;

    switch(k)
    {
    case 0:
        w0 = (1/6.f) * one_alpha2*one_alpha,
        w1 = (2/3.f) - 0.5f*alpha2*(2.0f-alpha),
        w2 = (1/6.f) -0.5f*one_alpha2*(2.0f-one_alpha),
        w3 = (1/6.f) * alpha*alpha2;
        break;
    case 1:
        w0 = -0.5f*alpha2 + alpha - 0.5f;
        w1 = 1.5f*alpha2 - 2.0f*alpha;
        w2 = -1.5f*alpha2 + alpha+0.5f;
        w3 = 0.5f*alpha2;
        break;
    case 2:
        w0 = 1.0f - alpha;
        w1 = 3.0f*alpha - 2.0f;
        w2 = -3.0f*alpha+1.0f;
        w3 = alpha;
        break;
    }
}




template <class S>
class bspline3_sampler
{
public:
    typedef S sampler_type;
    typedef typename S::result_type result_type;


    __device__ inline 
    result_type operator()(float2 coord_grid, int kx=0, int ky=0) const
    {
        // transform the coordinate from [0,extent] to [-0.5, extent-0.5]
    //    float2 coord_grid = make_float2(x-0.5f,y-0.5f);
        float2 index = make_float2(floor(coord_grid.x), floor(coord_grid.y));

        float2 alpha = coord_grid - index;

        float2 w0, w1, w2, w3;

        bspline3_weights(alpha.x, w0.x, w1.x, w2.x, w3.x, kx);
        bspline3_weights(alpha.y, w0.y, w1.y, w2.y, w3.y, ky);

        float2 g0 = w0 + w1,
               g1 = w2 + w3,
              // h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
               h0 = (w1 / g0) - 0.5f + index,
               h1 = (w3 / g1) + 1.5f + index;

        S sampler;

        // fetch the four linear
        result_type tex00 = sampler(h0.x, h0.y),
                    tex01 = sampler(h0.x, h1.y),
                    tex10 = sampler(h1.x, h0.y),
                    tex11 = sampler(h1.x, h1.y);

        // weigh along the y-direction
        if(ky == 0)
        {
            tex00 = lerp(tex01, tex00, g0.y);
            tex10 = lerp(tex11, tex10, g0.y);
        }
        else
        {
            tex00 = (tex00 - tex01)*g0.y;
            tex10 = (tex10 - tex11)*g0.y;
        }


        // weigh along the x-direction
        if(kx == 0)
            return lerp(tex10, tex00, g0.x);
        else
            return (tex10 - tex00)*g0.x;
    }
private:
};

#endif
