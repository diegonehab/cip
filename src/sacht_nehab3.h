#ifndef GPUFILTER_SACHT_NEHAB3_H
#define GPUFILTER_SACHT_NEHAB3_H

struct sacht_nehab3_weights
{
    template <class T>
    __device__ void operator()(T alpha, T &w0, T &w1, T &w2, T &w3, int k=0)
    {
        T alpha2 = alpha * alpha,
          alpha3 = alpha * alpha * alpha;

        switch(k)
        {
        case 0: // we are *not* implementing the discontinuities here
            w0 = 0.218848f - 0.497801f*alpha + 0.370818f*alpha2 - 0.0899247f*alpha3;
            w1 = 0.562591f + 0.0446542f*alpha - 0.700012f*alpha2 + 0.309387f*alpha3;
            w2 = 0.216621f + 0.427208f*alpha + 0.228149f*alpha2 - 0.309387f*alpha3;
            w3 = 0.00194006f + 0.0259387f*alpha + 0.101044f*alpha2 + 0.0899247f*alpha3;
            break;
        case 1: // this kernel is not differentiable
            w0 = 0.0f;
            w1 = 0.0f;
            w2 = 0.0f;
            w3 = 0.0f;
            break;
        case 2: // this kernel is not differentiable
            w0 = 0.0f;
            w1 = 0.0f;
            w2 = 0.0f;
            w3 = 0.0f;
            break;
        }
    }
};

// we are *not* implementing the discontinuities here
inline float sacht_nehab3(float r) 
{
    r = std::abs(r);

    if (r < 1.f)
        return 0.56259136f + 0.04465421f*r - 0.70001154f*r*r + 0.30938685f*r*r*r;
    else if (r < 2.f)
        return 1.17739188f - 1.50921205f*r + 0.64059253f*r*r - 0.08992473f*r*r*r;
    else 
        return 0.f;
}

// inline float sacht_nehab3(float r)
// {
//     r = std::abs(r);

//     if (r < 1.f) 
//         return (4.f + r*r*(-6.f + 3.f*r))/6.f;
//     else if (r < 2.f) 
//         return  (8.f + r*(-12.f + (6.f - r)*r))/6.f;
//     else 
//         return 0.f;
// }

#endif
