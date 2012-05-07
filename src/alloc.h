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
