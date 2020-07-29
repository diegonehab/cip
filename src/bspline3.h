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

#ifndef GPUFILTER_BSPLINE3_H
#define GPUFILTER_BSPLINE3_H

struct bspline3_weights
{
    template <class T>
    __device__ void operator()(T alpha, T &w0, T &w1, T &w2, T &w3, int k=0)
    {
        T one_alpha = 1.0f - alpha,
          alpha2 = alpha * alpha,
          one_alpha2 = one_alpha*one_alpha;

        switch(k)
        {
        case 0:
            w0 = (1/6.f) * one_alpha2*one_alpha,
            w1 = (2/3.f) - 0.5f*alpha2*(2.0f-alpha),
            w2 = (2/3.f) - 0.5f*one_alpha2*(2.0f-one_alpha),
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
};

inline float bspline3(float r)
{
    r = std::abs(r);

    if (r < 1.f) 
        return (4.f + r*r*(-6.f + 3.f*r))/6.f;
    else if (r < 2.f) 
        return  (8.f + r*(-12.f + (6.f - r)*r))/6.f;
    else 
        return 0.f;
}

#endif
