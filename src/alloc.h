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

#ifndef ALLOC_H
#define ALLOC_H

#include <stdexcept>
#include <sstream>
#include "error.h"


template <class T>
T *cuda_new(size_t elements)
{
    T *ptr = NULL;

    cudaMalloc((void **)&ptr, elements*sizeof(T));
    check_cuda_error("Memory allocation error");
    if(ptr == NULL)
        throw std::runtime_error("Memory allocation error");

    return ptr;
}

template <class T>
void cuda_delete(T *ptr)
{
    if(ptr == NULL)
        return;

    cudaFree((void *)ptr);
    check_cuda_error("Error freeing memory");
}

struct cuda_deleter
{
    template <class T>
    void operator()(T *ptr) const
    {
        cuda_delete(ptr);
    }
};

template <class T>
class cuda_allocator : public std::allocator<T>
{
public:
    typedef typename std::allocator<T>::pointer pointer;
    typedef typename std::allocator<T>::size_type size_type;

    pointer allocate(size_type n, std::allocator<void>::const_pointer hint=0)
    {
        return cuda_new<T>(n);
    }

    void deallocate(pointer ptr, size_type n)
    {
        cuda_delete(ptr);
    }

    void construct(pointer ptr, const T &val)
    {
        // do nothing
    }
    void destroy(pointer ptr)
    {
        // do nothing
    }
};


#endif
