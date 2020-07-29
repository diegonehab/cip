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

#ifndef RECFILTER_LINALG_H
#define RECFILTER_LINALG_H

#include <vector>

#if !defined(HOSTDEV)
#   if defined(__CUDA_ARCH__)
#       define HOSTDEV __host__ __device__
#   else
#       define HOSTDEV
#   endif
#endif

template <class T, int N>
class Vector 
{
public:
    std::vector<T> to_vector() const;

    HOSTDEV int size() const { return N; }

    HOSTDEV const T &operator[](int i) const;
    HOSTDEV T &operator[](int i);

    HOSTDEV operator const T *() const { return &m_data[0]; }
    HOSTDEV operator T *() { return &m_data[0]; }

    template <int R>
    HOSTDEV Vector<T,R> subv(int beg) const
    {
        assert(beg+R <= N);
        Vector<T,R> v;
        for(int i=beg; i<beg+R; ++i)
            v[i-beg] = m_data[i];
        return v;
    }

private:
    T m_data[N];
};


template <class T, int M, int N=M>
class Matrix
{
public:
    HOSTDEV int rows() const { return M; }
    HOSTDEV int cols() const { return N; }

    HOSTDEV const Vector<T,N> &operator[](int i) const;
    HOSTDEV Vector<T,N> &operator[](int i);

    HOSTDEV Vector<T,M> col(int j) const;
    HOSTDEV void set_col(int j, const Vector<T,M> &c);

private:
    Vector<T,N> m_data[M];
};

#include "linalg.hpp"

#endif
