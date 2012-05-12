#ifndef GUIFILTER_PIXEL_TRAITS_H
#define GUIFILTER_PIXEL_TRAITS_H

#include <stdlib.h> // cutil_math needs abs(int) when we're not using nvcc
#include <cutil_math.h> // for make_typeX

#ifndef HOSTDEV
#   define HOSTDEV __host__ __device__
#endif

template <class T, int D=1> 
struct pixel_traits {};

namespace detail/*{{{*/
{
    template <class T>
    struct pixel_traits_helper;

    template <>
    struct pixel_traits_helper<uchar1>
    {
        typedef unsigned char base_type;
        static const int planes = 1;
    };

    template <>
    struct pixel_traits_helper<uchar2>
    {
        typedef unsigned char base_type;
        static const int planes = 2;

    };

    template <>
    struct pixel_traits_helper<uchar3>
    {
        typedef unsigned char base_type;
        static const int planes = 3;

    };

    template <>
    struct pixel_traits_helper<uchar4>
    {
        typedef unsigned char base_type;
        static const int planes = 4;
    };

    template <>
    struct pixel_traits_helper<float1>
    {
        typedef float base_type;
        static const int planes = 1;
    };

    template <>
    struct pixel_traits_helper<float2>
    {
        typedef float base_type;
        static const int planes = 2;
    };

    template <>
    struct pixel_traits_helper<float3>
    {
        typedef float base_type;
        static const int planes = 3;
    };

    template <>
    struct pixel_traits_helper<float4>
    {
        typedef float base_type;
        static const int planes = 4;
    };


    template <class T, class B, int C>
    struct members_helper;

    template <class T, class B>
    struct members_helper<T,B,1>
    {
        HOSTDEV static T make(T x)
        {
            return x;
        }
        template <class U>
        HOSTDEV static T make(U v)
        {
            return v.x;
        }

        HOSTDEV
        static void assign(B *pix, int stride, const T &data)
        {
            *pix = data.x;
        }
        HOSTDEV
        static void assign(T &data, const B *pix, int stride)
        {
            data.x = *pix;
        }

        HOSTDEV
        static void assign(B *pix, int stride, const B &data)
        {
            *pix = data;
        }
        HOSTDEV
        static void assign(B &data, const B *pix, int stride)
        {
            data = *pix;
        }
    };

    template <class T, class B>
    struct members_helper<T,B,2>
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
    struct members_helper<T,B,3>
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
    struct members_helper<T,B,4>
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
struct pixel_traits<T,1> /*{{{*/
    : detail::pixtraits_parent<T>::type
{ 
private:
public:
    typedef typename detail::pixtraits_parent<T>::type::type type;

    HOSTDEV
    static void assign(type *pix, int stride, const type &data)
    {
        *pix = data;
    }
    HOSTDEV
    static void assign(type &data, const type *pix, int stride)
    {
        data = *pix;
    }
};/*}}}*/

template <class T> 
struct pixel_traits<const T,1> /*{{{*/
    : pixel_traits<T,1>
{ 
};/*}}}*/

template <> 
struct pixel_traits<float,1> /*{{{*/
    : detail::members_helper<float1,float,1>
{ 
    typedef float type; 
    typedef float texel_type;
    typedef type base_type;
    static const int planes = 1;
};/*}}}*/

template <> 
struct pixel_traits<float,2> /*{{{*/
    : detail::members_helper<float2,float,2>
{ 
    typedef float2 type; 
    typedef float2 texel_type;
    typedef float base_type;
    static const int planes = 2;

    template <class U>
    HOSTDEV static type make(U x)
    {
        return make_float2(x);
    }
    template <class U, class V>
    HOSTDEV static type make(U x, V y)
    {
        return make_float2(x,y);
    }
};/*}}}*/

template <> 
struct pixel_traits<float,3> /*{{{*/
    : detail::members_helper<float3,float,3>
{ 
    typedef float3 type; 
    typedef float4 texel_type;
    typedef float base_type;
    static const int planes = 3;

    template <class U>
    HOSTDEV static type make(U x)
    {
        return make_float3(x);
    }
    template <class U, class V>
    HOSTDEV static type make(U x, V y)
    {
        return make_float3(x,y);
    }

    template <class U, class V, class W>
    HOSTDEV static type make(U x, V y, W z)
    {
        return make_float3(x,y,z);
    }
};/*}}}*/

template <> 
struct pixel_traits<float,4> /*{{{*/
    : detail::members_helper<float4,float,4>
{ 
    typedef float4 type; 
    typedef float4 texel_type;
    typedef float base_type;
    static const int planes = 4;

    template <class U>
    HOSTDEV static type make(U x)
    {
        return make_float4(x);
    }
    template <class U, class V>
    HOSTDEV static type make(U x, V y)
    {
        return make_float4(x,y);
    }

    template <class U, class V, class W>
    HOSTDEV static type make(U x, V y, W z)
    {
        return make_float4(x,y,z);
    }

    template <class U, class V, class W, class X>
    HOSTDEV static type make(U x, V y, W z, X w)
    {
        return make_float4(x,y,z,w);
    }
};/*}}}*/


template <> 
struct pixel_traits<unsigned char,1> /*{{{*/
    : detail::members_helper<uchar1,unsigned char,1>
{ 
    typedef uchar1 type; 
    typedef uchar1 texel_type;
    typedef unsigned char base_type;
    static const int planes = 1;
};/*}}}*/

template <> 
struct pixel_traits<unsigned char,2> /*{{{*/
    : detail::members_helper<uchar2,unsigned char,2>
{ 
    typedef uchar2 type; 
    typedef uchar2 texel_type;
    typedef unsigned char base_type;
    static const int planes = 2;

    template <class U>
    HOSTDEV static type make(U x)
    {
        return make_uchar2(x);
    }
    template <class U, class V>
    HOSTDEV static type make(U x, V y)
    {
        return make_uchar2(x,y);
    }
};/*}}}*/

template <> 
struct pixel_traits<unsigned char,3> /*{{{*/
    : detail::members_helper<uchar3,unsigned char,3>
{ 
    typedef uchar3 type; 
    typedef uchar4 texel_type;
    typedef unsigned char base_type;
    static const int planes = 3;

    template <class U>
    HOSTDEV static type make(U x)
    {
        return make_uchar3(x);
    }
    template <class U, class V>
    HOSTDEV static type make(U x, V y)
    {
        return make_uchar3(x,y);
    }

    template <class U, class V, class W>
    HOSTDEV static type make(U x, V y, W z)
    {
        return make_uchar3(x,y,z);
    }
};/*}}}*/

template <> 
struct pixel_traits<unsigned char,4> /*{{{*/
    : detail::members_helper<uchar4,unsigned char,4>
{ 
    typedef uchar4 type; 
    typedef uchar4 texel_type;
    typedef unsigned char base_type;
    static const int planes = 4;

    template <class U>
    HOSTDEV static type make(U x)
    {
        return make_uchar4(x);
    }
    template <class U, class V>
    HOSTDEV static type make(U x, V y)
    {
        return make_uchar4(x,y);
    }

    template <class U, class V, class W>
    HOSTDEV static type make(U x, V y, W z)
    {
        return make_uchar4(x,y,z);
    }

    template <class U, class V, class W, class X>
    HOSTDEV static type make(U x, V y, W z, X w)
    {
        return make_uchar4(x,y,z,w);
    }
};/*}}}*/


#endif
