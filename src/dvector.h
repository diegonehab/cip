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

/**
 *  @file dvector.h
 *  @brief Device Vector Class definition
 *  @author Rodolfo Lima
 *  @date February, 2011
 */

#ifndef DVECTOR_H
#define DVECTOR_H


//== INCLUDES =================================================================

#include <vector>

#include "alloc.h"

//== CLASS DEFINITION =========================================================

/** @class dvector dvector.hh
 *  @brief Device Vector class
 *
 *  Device vector is a STL-based vector in the GPU memory.
 *
 *  @tparam T Device vector values type
 */
template <class T>
class dvector/*{{{*/
{
public:
    /** Constructor
     *  @param[in] data Vector gpu data to be copied into this object
     *  @param[in] size Vector data size
     */
    dvector(const T *data, size_t size)
        : m_size(size), m_capacity(size), m_data(data)
    {
    }

    /** Copy Constructor
     *  @param[in] that Copy that object to this object
     */
    dvector(const dvector &that) 
        : m_size(0), m_capacity(0), m_data(NULL)
    {
        *this = that;
    }

    /// Default constructor
    dvector(size_t size=0) 
        : m_size(0), m_capacity(0), m_data(NULL)
    {
        resize(size);
    }

    /// Destructor
    ~dvector()
    {
        cuda_delete(m_data);
        m_data = 0;
        m_capacity = 0;
        m_size = 0;
    }
    
    /** @brief Resize this vector
     *  @param[in] size The new vector size
     */
    void resize(size_t size)
    {
        if(size > m_capacity)
        {
            cuda_delete(m_data);
            m_data = 0;
            m_capacity = 0;
            m_size = 0;

            m_data = cuda_new<T>(size);
            m_capacity = size;
            m_size = size;
        }
        else
            m_size = size;
    }

    void reset(T *data, size_t size)
    {
        cuda_delete(m_data);
        m_data = NULL;

        m_capacity = m_size = size;
        m_data = data;
    }

    /** @brief Clear this vector
     */
    void clear()
    {
        m_size = 0;
    }

    void fillzero()
    {
        cudaMemset(m_data, 0, m_size*sizeof(T));
    }

    /** @brief Read/write operator
     *
     *  @param idx Index of vector value
     *  @return Vector value at index
     */
    T &operator[](int idx) const
    {
        static T value;
        cudaMemcpy(&value, data()+idx, sizeof(T), cudaMemcpyDeviceToHost);
        return value;
    }

    /** @brief Assign operator
     *  @param that Device vector to copy from
     *  @return This device vector with assigned values
     */
    dvector &operator=(const dvector &that)
    {
        resize(that.size());
        cudaMemcpy(data(), that.data(), size()*sizeof(T), cudaMemcpyDeviceToDevice);
        check_cuda_error("Error during memcpy from device to device");
        return *this;
    }

    /** @brief Copy values from this vector to a host (CPU) vector
     *  @param[out] data Host Vector to copy values to
     *  @param[in] s Maximum number of elements to copy
     */
    void copy_to_host(T *data, size_t s) const
    {
        using std::min;

        cudaMemcpy(data, this->data(), min(size(),s)*sizeof(T), cudaMemcpyDeviceToHost);
        check_cuda_error("Error during memcpy from device to host");
    }

    void copy_from_host(const T *data, size_t s)
    {
        using std::min;

        resize(s);

        cudaMemcpy(this->data(), data, s*sizeof(T), cudaMemcpyHostToDevice);
        check_cuda_error("Error during memcpy from device to host");
    }

    void copy2D_to_host(T *out, size_t width, size_t height, size_t rowstride) const
    {
        cudaMemcpy2D(out, width*sizeof(T), 
                     data(), rowstride*sizeof(T), 
                     width*sizeof(T), height, cudaMemcpyDeviceToHost);

        check_cuda_error("Error during memcpy from device to host");
    }

    void copy2D_from_host(const T *in, size_t width, size_t height, size_t rowstride)
    {
        resize(rowstride*height);

        cudaMemcpy2D(data(), rowstride*sizeof(T), in, width*sizeof(T), 
                     width*sizeof(T), height, cudaMemcpyHostToDevice);

        check_cuda_error("Error during memcpy from host to device");
    }

    /** @brief Check if this vector is empty
     *  @return True if this vector is empty
     */
    bool empty() const { return size()==0; }

    /** @brief Size of this vector
     *  @return Vector size
     */
    size_t size() const { return m_size; }

    /** @brief Data in this vector
     *  @return Vector data
     */
    T *data() { return m_data; }

    /** @overload const T *data() const
     *  @return Constant vector data
     */
    const T *data() const { return m_data; }

    T &back() const { return operator[](size()-1); }

    /** @brief Data in this vector
     *  @return Vector data
     */
    operator T*() { return data(); }

    /** @overload operator const T*() const
     *  @return Constant vector data
     */
    operator const T*() const { return data(); }


    const T *operator&() const { return m_data; }
    T *operator&() { return m_data; }

    /** @brief Swap vector values
     *  @param [in,out] a Vector to be swapped
     *  @param [in,out] b Vector to be swapped
     */
    friend void swap(dvector &a, dvector &b)
    {
        using std::swap;

        swap(a.m_data, b.m_data);
        swap(a.m_size, b.m_size);
        swap(a.m_capacity, b.m_capacity);
    }

private:
    T *m_data; ///< Vector data
    size_t m_size, m_capacity; ///< Vector size and capacity
};/*}}}*/

template <class T>
class dvector_ptr/*{{{*/
{
public:
    __host__ __device__
    dvector_ptr(T *data, size_t size)
        : m_size(size), m_data(data)
    {
    }

    __host__ __device__
    void reset(T *data, size_t size)
    {
        m_size = size;
        m_data = data;
    }

    void fillzero()
    {
        cudaMemset(m_data, 0, m_size*sizeof(T));
    }

#if __CUDA_ARCH__
    T &operator[](int idx) const
    {
        return data()[idx];
    }
#else
    T &operator[](int idx) const
    {
        static T value;
        cudaMemcpy(&value, data()+idx, sizeof(T), cudaMemcpyDeviceToHost);
        return value;
    }
#endif

    size_t copy_to_host(T *data, size_t s=UINT_MAX) const
    {
        using std::min;

        size_t nwritten = min(size(),s);

        cudaMemcpy(data, this->data(), nwritten*sizeof(T), cudaMemcpyDeviceToHost);
        check_cuda_error("Error during memcpy from device to host");
        return nwritten;
    }

    size_t copy_from_host(const T *data, size_t s=UINT_MAX)
    {
        using std::min;

        size_t nwritten = min(size(),s);

        cudaMemcpy(this->data(), data, nwritten*sizeof(T), cudaMemcpyHostToDevice);
        check_cuda_error("Error during memcpy from device to host");
        return nwritten;
    }

    /** @brief Check if this vector is empty
     *  @return True if this vector is empty
     */
    __device__ __host__
    bool empty() const { return size()==0; }

    /** @brief Size of this vector
     *  @return Vector size
     */
    __device__ __host__
    size_t size() const { return m_size; }

    /** @brief Data in this vector
     *  @return Vector data
     */
    __device__ __host__
    T *data() { return m_data; }

    /** @overload const T *data() const
     *  @return Constant vector data
     */
    __device__ __host__
    const T *data() const { return m_data; }

    __device__ __host__
    T &back() const { return operator[](size()-1); }

    /** @brief Data in this vector
     *  @return Vector data
     */
    __device__ __host__
    operator T*() { return data(); }

    /** @overload operator const T*() const
     *  @return Constant vector data
     */
    __device__ __host__
    operator const T*() const { return data(); }


    __device__ __host__
    const T *operator&() const { return m_data; }

    __device__ __host__
    T *operator&() { return m_data; }

    /** @brief Swap vector values
     *  @param [in,out] a Vector to be swapped
     *  @param [in,out] b Vector to be swapped
     */
    __device__ __host__
    friend void swap(dvector_ptr &a, dvector_ptr &b)
    {
        using std::swap;

        swap(a.m_data, b.m_data);
        swap(a.m_size, b.m_size);
    }

private:
    T *m_data; ///< Vector data
    size_t m_size; ///< Vector size and capacity
};/*}}}*/

//=== IMPLEMENTATION ==========================================================

/** @relates dvector
 *  @brief Copy to the CPU a vector in the GPU
 *
 *  This function copies a device vector (GPU) to a host vector (CPU).
 *
 *  @param[in] d_vec Pointer to the device vector (in the GPU memory)
 *  @param[in] len Length of the device vector
 *  @return Host vector (in the CPU memory) as a STL vector
 *  @tparam T Vector values type
 */
template <class T>
std::vector<T> to_host(const T *d_vec, unsigned len)
{
    std::vector<T> out;
    out.resize(len);

    cudaMemcpy(&out[0], d_vec, len*sizeof(T), cudaMemcpyDeviceToHost);
    check_cuda_error("Error during memcpy from device to host");

    return out;
}

/** @relates dvector
 *  @overload
 *
 *  @param[in] v Device vector (in the GPU memory)
 *  @return Host vector (in the CPU memory) as a STL vector
 *  @tparam T Vector values type
 */
template <class T>
std::vector<T> to_host(const dvector<T> &v)
{
    return to_host(v.data(), v.size());
}

//=============================================================================
#endif // DVECTOR_H
//=============================================================================
//vi: ai sw=4 ts=4
