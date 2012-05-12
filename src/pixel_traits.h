#ifndef GUIFILTER_PIXEL_TRAITS_H
#define GUIFILTER_PIXEL_TRAITS_H

#include <stdlib.h> // cutil_math needs abs(int) when we're not using nvcc
#include <cutil_math.h> // for make_typeX
#include "util.h"

#ifndef HOSTDEV
#   define HOSTDEV __host__ __device__
#endif

HOSTDEV
inline uchar3 make_uchar3(const uchar4 &v)
{
    return make_uchar3(v.x,v.y,v.z);
}

HOSTDEV
inline uchar4 make_uchar4(const uchar3 &v)
{
    return make_uchar4(v.x,v.y,v.z,1);
}

template <class T, int D=1, class EN=void> 
struct pixel_traits;

namespace detail/*{{{*/
{
    template <class T>
    struct pixel_traits_helper
    {
        typedef T base_type;
        static const int planes = 1;
        static const bool is_composite = false;
    };

    template <>
    struct pixel_traits_helper<uchar2>
    {
        typedef unsigned char base_type;
        static const int planes = 2;
        static const bool is_composite = true;

    };

    template <>
    struct pixel_traits_helper<uchar3>
    {
        typedef unsigned char base_type;
        static const int planes = 3;
        static const bool is_composite = true;

    };

    template <>
    struct pixel_traits_helper<uchar4>
    {
        typedef unsigned char base_type;
        static const int planes = 4;
        static const bool is_composite = true;
    };

    template <>
    struct pixel_traits_helper<float2>
    {
        typedef float base_type;
        static const int planes = 2;
        static const bool is_composite = true;
    };

    template <>
    struct pixel_traits_helper<float3>
    {
        typedef float base_type;
        static const int planes = 3;
        static const bool is_composite = true;
    };

    template <>
    struct pixel_traits_helper<float4>
    {
        typedef float base_type;
        static const int planes = 4;
        static const bool is_composite = true;
    };


    template <class T, class B, int C>
    struct assign_helper;

    template <class T, class B>
    struct assign_helper<T,B,1>
    {
        template <class U>
        HOSTDEV
        static void assign(B *pix, int stride, const U &data)
        {
            *pix = pixel_traits<B>::make_pixel(data);
        }
        template <class U>
        HOSTDEV
        static void assign(U &data, const B *pix, int stride)
        {
            data = pixel_traits<U>::make_pixel(*pix);
        }
    };

    template <class T, class B>
    struct assign_helper<T,B,2>
    {
        HOSTDEV
        static void assign(B *pix, int stride, const T &data)
        {
            pix[0] = data.x;
            pix[stride] = data.y;
        }
        HOSTDEV
        static void assign(T &data, const B *pix, int stride)
        {
            data.x = pix[0];
            data.y = pix[stride];
        }
    };

    template <class T, class B>
    struct assign_helper<T,B,3>
    {
        HOSTDEV
        static void assign(B *pix, int stride, const T &data)
        {
            pix[0] = data.x;
            pix[stride] = data.y;
            pix[stride*2] = data.z;
        }
        HOSTDEV
        static void assign(T &data, const B *pix, int stride)
        {
            data.x = pix[0];
            data.y = pix[stride];
            data.z = pix[stride*2];
        }
    };

    template <class T, class B>
    struct assign_helper<T,B,4>
    {
        HOSTDEV
        static void assign(B *pix, int stride, const T &data)
        {
            pix[0] = data.x;
            pix[stride] = data.y;
            pix[stride*2] = data.z;
            pix[stride*3] = data.w;
        }
        HOSTDEV
        static void assign(T &data, const B *pix, int stride)
        {
            data.x = pix[0];
            data.y = pix[stride];
            data.z = pix[stride*2];
            data.w = pix[stride*3];
        }
    };

    template <class T>
    struct pixtraits_parent
    {
    private:
        typedef typename pixel_traits_helper<T>::base_type base_type;
        static const int planes = pixel_traits_helper<T>::planes;
    public:
        typedef pixel_traits<base_type, planes> type;
    };
};/*}}}*/

template <class T> 
struct pixel_traits<T,1,/*{{{*/
    typename enable_if<detail::pixel_traits_helper<T>::is_composite &&
                       !is_const<T>::value && !is_volatile<T>::value>::type>
    : detail::pixtraits_parent<T>::type
{ 
private:
    typedef typename detail::pixtraits_parent<T>::type base;
public:
    typedef typename base::pixel_type pixel_type;
    typedef typename base::base_type base_type;
    static const int planes = base::planes;

    HOSTDEV static T make_pixel(T x)
    {
        return x;
    }

    template <class U>
    HOSTDEV static T make_pixel(U v)
    {
        return pixel_traits<base_type,planes>::make_pixel(v);
    }

    template <class U>
    HOSTDEV
    static void assign(pixel_type *pix, int stride, const U &data)
    {
        *pix = make_pixel(data);
    }

    template <class U>
    HOSTDEV
    static void assign(U &data, const pixel_type *pix, int stride)
    {
        data = pixel_traits<U>::make_pixel(*pix);
    }
};/*}}}*/

template <class T> 
struct pixel_traits<T,1,/*{{{*/
    typename enable_if<!detail::pixel_traits_helper<T>::is_composite &&
                       !is_const<T>::value && !is_volatile<T>::value>::type>
    : detail::assign_helper<float,float,1>
{ 
    typedef T pixel_type; 
    typedef T texel_type;
    typedef pixel_type base_type;
    static const int planes = 1;

    HOSTDEV static pixel_type make_pixel(base_type x)
    {
        return x;
    }

    template <class U>
    HOSTDEV static 
    typename enable_if<pixel_traits<U>::is_composite,pixel_type>::type
        make_pixel(U v)
    {
        return v.x;
    }
};/*}}}*/

template <class T, int C, class EN> 
struct pixel_traits<const T,C,EN> /*{{{*/
    : pixel_traits<T,C>
{ 
};/*}}}*/

template <class T, int C, class EN> 
struct pixel_traits<volatile T,C,EN> /*{{{*/
    : pixel_traits<T,C>
{ 
};/*}}}*/

template <class T, int C, class EN> 
struct pixel_traits<const volatile T,C,EN> /*{{{*/
    : pixel_traits<T,C>
{ 
};/*}}}*/

#if 0
// these are for disambiguation
template <class T, class EN> 
struct pixel_traits<const T,1,EN> /*{{{*/
    : pixel_traits<T>
{ 
};/*}}}*/

template <class T, class EN> 
struct pixel_traits<volatile T,1,EN> /*{{{*/
    : pixel_traits<T>
{ 
};/*}}}*/

template <class T, class EN> 
struct pixel_traits<const volatile T,1,EN> /*{{{*/
    : pixel_traits<T>
{ 
};/*}}}*/
#endif

template <> 
struct pixel_traits<float,2> /*{{{*/
    : detail::assign_helper<float2,float,2>
{ 
    typedef float2 pixel_type; 
    typedef float2 texel_type;
    typedef float base_type;
    static const int planes = 2;

    template <class U>
    HOSTDEV static pixel_type make_pixel(U x)
    {
        return make_float2(x);
    }
    template <class U, class V>
    HOSTDEV static pixel_type make_pixel(U x, V y)
    {
        return make_float2(x,y);
    }
};/*}}}*/

template <> 
struct pixel_traits<float,3> /*{{{*/
    : detail::assign_helper<float3,float,3>
{ 
    typedef float3 pixel_type; 
    typedef float4 texel_type;
    typedef float base_type;
    static const int planes = 3;

    template <class U>
    HOSTDEV static pixel_type make_pixel(U x)
    {
        return make_float3(x);
    }
    template <class U, class V>
    HOSTDEV static pixel_type make_pixel(U x, V y)
    {
        return make_float3(x,y);
    }

    template <class U, class V, class W>
    HOSTDEV static pixel_type make_pixel(U x, V y, W z)
    {
        return make_float3(x,y,z);
    }
};/*}}}*/

template <> 
struct pixel_traits<float,4> /*{{{*/
    : detail::assign_helper<float4,float,4>
{ 
    typedef float4 pixel_type; 
    typedef float4 texel_type;
    typedef float base_type;
    static const int planes = 4;

    template <class U>
    HOSTDEV static pixel_type make_pixel(U x)
    {
        return make_float4(x);
    }
    template <class U, class V>
    HOSTDEV static pixel_type make_pixel(U x, V y)
    {
        return make_float4(x,y);
    }

    template <class U, class V, class W>
    HOSTDEV static pixel_type make_pixel(U x, V y, W z)
    {
        return make_float4(x,y,z);
    }

    template <class U, class V, class W, class X>
    HOSTDEV static pixel_type make_pixel(U x, V y, W z, X w)
    {
        return make_float4(x,y,z,w);
    }
};/*}}}*/


template <> 
struct pixel_traits<unsigned char,2> /*{{{*/
    : detail::assign_helper<uchar2,unsigned char,2>
{ 
    typedef uchar2 pixel_type; 
    typedef uchar2 texel_type;
    typedef unsigned char base_type;
    static const int planes = 2;

    template <class U>
    HOSTDEV static pixel_type make_pixel(U x)
    {
        return make_uchar2(x);
    }
    template <class U, class V>
    HOSTDEV static pixel_type make_pixel(U x, V y)
    {
        return make_uchar2(x,y);
    }
};/*}}}*/

template <> 
struct pixel_traits<unsigned char,3> /*{{{*/
    : detail::assign_helper<uchar3,unsigned char,3>
{ 
    typedef uchar3 pixel_type; 
    typedef uchar4 texel_type;
    typedef unsigned char base_type;
    static const int planes = 3;

    template <class U>
    HOSTDEV static pixel_type make_pixel(U x)
    {
        return make_uchar3(x);
    }
    template <class U, class V>
    HOSTDEV static pixel_type make_pixel(U x, V y)
    {
        return make_uchar3(x,y);
    }

    template <class U, class V, class W>
    HOSTDEV static pixel_type make_pixel(U x, V y, W z)
    {
        return make_uchar3(x,y,z);
    }
};/*}}}*/

template <> 
struct pixel_traits<unsigned char,4> /*{{{*/
    : detail::assign_helper<uchar4,unsigned char,4>
{ 
    typedef uchar4 pixel_type; 
    typedef uchar4 texel_type;
    typedef unsigned char base_type;
    static const int planes = 4;

    template <class U>
    HOSTDEV static pixel_type make_pixel(U x)
    {
        return make_uchar4(x);
    }
    template <class U, class V>
    HOSTDEV static pixel_type make_pixel(U x, V y)
    {
        return make_uchar4(x,y);
    }

    template <class U, class V, class W>
    HOSTDEV static pixel_type make_pixel(U x, V y, W z)
    {
        return make_uchar4(x,y,z);
    }

    template <class U, class V, class W, class X>
    HOSTDEV static pixel_type make_pixel(U x, V y, W z, X w)
    {
        return make_uchar4(x,y,z,w);
    }
};/*}}}*/

template <class T>
struct texel_traits;

template <>
struct texel_traits<float>/*{{{*/
    : pixel_traits<float>
{
    HOSTDEV static float make_texel(float x)
    {
        return x;
    }

    template <class U>
    HOSTDEV static float make_texel(U v)
    {
        return v.x;
    }
};/*}}}*/

template <>
struct texel_traits<float2>/*{{{*/
    : pixel_traits<float2>
{
    template <class U>
    HOSTDEV static float2 make_texel(U x)
    {
        return make_float2(x);
    }
    template <class U, class V>
    HOSTDEV static float2 make_texel(U x, V y)
    {
        return make_float2(x,y);
    }
};/*}}}*/

template <>
struct texel_traits<float4>/*{{{*/
    : pixel_traits<float3>
{
    template <class U>
    HOSTDEV static float4 make_texel(U x)
    {
        return make_float4(x);
    }
    template <class U, class V>
    HOSTDEV static float4 make_texel(U x, V y)
    {
        return make_float4(x,y);
    }

    template <class U, class V, class W>
    HOSTDEV static float4 make_texel(U x, V y, W z)
    {
        return make_float4(x,y,z);
    }

    template <class U, class V, class W, class X>
    HOSTDEV static float4 make_texel(U x, V y, W z, X w)
    {
        return make_float4(x,y,z,w);
    }
};/*}}}*/

#endif
