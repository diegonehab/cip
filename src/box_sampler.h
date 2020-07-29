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

#ifndef NLFILTER_BOX_SAMPLER_H
#define NLFILTER_BOX_SAMPLER_H

#include "math_util.h"

template <class S>
class box_sampler
{
public:
    typedef S sampler_type;
    typedef typename S::result_type result_type;

    template <class S2>
    struct rebind_sampler
    {
        typedef box_sampler<S2> type;
    };

    __device__ inline 
    result_type operator()(float2 pos, int kx=0, int ky=0) const
    {
        S sampler;

        if(kx == 0 && ky == 0)
            return sampler(pos.x, pos.y);
        else
        {
            // TODO: we must do something sensible here,
            // box derivatives are 0!
            return pixel_traits<result_type>::make_pixel(0);
        }
    }
};

#endif
