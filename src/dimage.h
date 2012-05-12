#ifndef NLFILTER_DIMAGE_H
#define NLFILTER_DIMAGE_H

#include <cassert>
#include "dvector.h"
#include "util.h"
#include "pixel_traits.h"

template <class T, int C=1>
class dimage_ptr;

template <class T, int C=1>
class dimage
{
public:
    typedef typename pixel_traits<T,1>::texel_type texel_type;

    dimage(texel_type *data, int width, int height, int rowstride)
        : m_data(data, rowstride*height)
        , m_width(width)
        , m_height(height)
        , m_rowstride(rowstride)
    {
        
    }

    dimage(int width, int height, int rowstride=0)
    {
        resize(width, height, rowstride);
    }

    dimage() : m_width(0), m_height(0), m_rowstride(0)
    {
    }

    void reset(texel_type *data, int width, int height, int rowstride)
    {
        m_width = width;
        m_height = height;
        if(rowstride == 0)
            m_rowstride = ((m_width + 256-1)/256)*256; // multiple of 256
        else
        {
            m_rowstride = rowstride;
            if(rowstride < width)
                throw std::runtime_error("Bad row stride");
        }

        m_data.reset(data, m_rowstride*m_height*C);
    }

    void copy_to_host(texel_type *out) const
    {
        for(int i=0; i<C; ++i)
        {
            cudaMemcpy2D(out+width()*height()*i, width()*sizeof(texel_type), 
                         &m_data+i*channelstride(), 
                         rowstride()*sizeof(texel_type), 
                         width()*sizeof(texel_type), height(), 
                         cudaMemcpyDeviceToHost);
        }

        check_cuda_error("Error during memcpy from device to host");
    }

    void copy_to_host(std::vector<texel_type> &out) const
    {
        out.resize(width()*height());
        copy_to_host(&out[0]);
    }

    void copy_from_host(const texel_type *in, int w, int h, int rs=0)
    {
        resize(w, h, rs);

        for(int i=0; i<C; ++i)
        {
            cudaMemcpy2D(&m_data+i*channelstride(), rowstride()*sizeof(texel_type), 
                         in+i*width()*height(), width()*sizeof(texel_type), 
                         width()*sizeof(texel_type), height(), cudaMemcpyHostToDevice);
        }

        check_cuda_error("Error during memcpy from host to device");
    }
    void copy_from_host(const std::vector<texel_type> &in, 
                        int width, int height, int rowstride=0)
    {
        assert(in.size() == width*height);
        copy_from_host(&in[0], width, height, rowstride);
    }


    void resize(int width, int height, int rowstride=0)
    {
        m_width = width;
        m_height = height;
        if(rowstride == 0)
            m_rowstride = ((m_width + 256-1)/256)*256; // multiple of 256
        else
        {
            if(rowstride < m_width)
                throw std::runtime_error("Bad row stride");
            m_rowstride = rowstride;
        }

        m_data.resize(m_rowstride*m_height*C);

    }

    bool empty() const
    {
        return m_data.empty();
    }

    dimage &operator=(const dimage_ptr<const texel_type, C> &img)
    {
        resize(img.width(), img.height(), img.rowstride());

        for(int i=0; i<C; ++i)
        {
            cudaMemcpy2D(&m_data+channelstride(),rowstride()*sizeof(texel_type), 
                         &img+i*img.channelstride(), rowstride()*sizeof(texel_type), 
                         width()*sizeof(texel_type), height(), cudaMemcpyDeviceToDevice);
        }

        return *this;
    }

    dimage &operator=(const dimage &that)
    {
        m_data = that.m_data;

        m_width = that.m_width;
        m_height = that.m_height;
        m_rowstride = that.m_rowstride;
        return *this;
    }

    int width() const { return m_width; }
    int height() const { return m_height; }
    int rowstride() const { return m_rowstride; }
    int channelstride() const { return rowstride()*height(); } 

    dimage_ptr<T,C> operator&();
    dimage_ptr<const T,C> operator&() const;

    operator texel_type*() { return m_data; }
    operator const texel_type*() const { return m_data; }

    int offset_at(int x, int y) const { return y*rowstride()+x; }
    bool is_inside(int x, int y) const 
        { return x < width() && y < height(); }

    dimage_ptr<T,1> operator[](int i);

    dimage_ptr<const T, 1> operator[](int i) const;

private:
    dvector<texel_type> m_data;
    int m_width, m_height, m_rowstride;
};

template <class T, int C>
class dimage_ptr
{
public:
    typedef typename copy_const<T,typename pixel_traits<T,1>::texel_type>::type texel_type;
private:

    template <int D, class EN = void>
    class pixel_proxy/*{{{*/
    {
        dimage_ptr &m_img;

    public:
        HOSTDEV
        pixel_proxy(dimage_ptr &img): m_img(img) {}

        typedef typename pixel_traits<T,C>::pixel_type value_type;

        HOSTDEV
        pixel_proxy &operator=(const pixel_proxy &p)
        {
            pixel_traits<texel_type,D>::assign(m_img.m_data, m_img.channelstride(), p);
            return *this;
        }

        HOSTDEV
        pixel_proxy &operator=(const value_type &v)
        {
            pixel_traits<texel_type,D>::assign(m_img.m_data, m_img.channelstride(), v);
            return *this;
        }

        HOSTDEV
        pixel_proxy &operator+=(const value_type &v)
        {
            value_type temp;
            pixel_traits<texel_type,D>::assign(temp, m_img.m_data, m_img.channelstride());
            temp += v;
            pixel_traits<texel_type,D>::assign(m_img.m_data, m_img.channelstride(), temp);
            return *this;
        }

        HOSTDEV
        pixel_proxy &operator-=(const value_type &v)
        {
            value_type temp;
            pixel_traits<texel_type,D>::assign(temp, m_img.m_data, m_img.channelstride());
            temp -= v;
            pixel_traits<texel_type,D>::assign(m_img.m_data, m_img.channelstride(), temp);
            return *this;
        }

        HOSTDEV
        operator value_type() const
        {
            value_type val;
            pixel_traits<texel_type,D>::assign(val, m_img.m_data, m_img.channelstride());
            return val;
        }

        T *operator&() { return m_img.m_data; }
        const T *operator&() const { return m_img.m_data; }
    };/*}}}*/

    template <int D>
    class pixel_proxy<D, typename enable_if<(D<=0 || D>=5)>::type>
    {
    public:
        HOSTDEV
        pixel_proxy(dimage_ptr &img) {}
    };

    template <int D, class EN = void>
    class const_pixel_proxy/*{{{*/
    {
        const dimage_ptr &m_img;

    public:
        HOSTDEV
        const_pixel_proxy(const dimage_ptr &img) : m_img(img) {}

        typedef typename pixel_traits<T,C>::pixel_type value_type;

        HOSTDEV
        operator value_type() const
        {
            value_type val;
            pixel_traits<T,D>::assign(val, m_img.m_data, m_img.channelstride());
            return val;
        }
        const T *operator&() const { return m_img.m_data; }
    };/*}}}*/

    template <int D>
    class const_pixel_proxy<D, typename enable_if<(D<=0 || D>=5)>::type>
    {
    public:
        HOSTDEV
        const_pixel_proxy(const dimage_ptr &img) {}
    };

public:
    HOSTDEV
    dimage_ptr(texel_type *data, int width, int height, int rowstride)
        : m_width(width), m_height(height), m_rowstride(rowstride)
        , m_data(data)
    {
    }

    template <class P>
    HOSTDEV
    dimage_ptr(const dimage_ptr<P,C> &that,
               typename enable_if<is_convertible<P,T>::value>::type* =NULL)
        : m_width(that.m_width), m_height(that.m_height)
        , m_rowstride(that.m_rowstride)
        , m_data(that.m_data)
    {
    }

    HOSTDEV
    int width() const { return m_width; }
    HOSTDEV
    int height() const { return m_height; }
    HOSTDEV
    int rowstride() const { return m_rowstride; }

    HOSTDEV
    int offset_at(int x, int y) const
        { return y*rowstride()+x; }

    HOSTDEV
    bool is_inside(int x, int y) const
        { return x < width() && y < height(); }

    dimage_ptr &operator=(dimage_ptr<const T,C> img)
    {
        if(width() != img.width() || height() != img.height())
            throw std::runtime_error("Image dimensions don't match");

        for(int i=0; i<C; ++i)
        {
            cudaMemcpy2D(&m_data+channelstride(), width()*sizeof(texel_type), 
                         &img+i*img.channelstride(), 
                         rowstride()*sizeof(texel_type), 
                         width()*sizeof(texel_type), height(), 
                         cudaMemcpyDeviceToDevice);
        }
        return *this;
    }

    dimage_ptr &operator=(const dimage<T,C> &img)
    {
        return *this = &img;
    }

    HOSTDEV
    dimage_ptr<T,1> operator[](int i)
    {
        return dimage_ptr<T,1>(&m_data[i*channelstride()], 
                               width(), height(), rowstride());
    }

    HOSTDEV
    dimage_ptr<const T,1> operator[](int i) const
    {
        return dimage_ptr<const T,1>(&m_data[i*channelstride()], 
                                     width(), height(), rowstride());
    }

    HOSTDEV
    operator T*() { return m_data; }

    HOSTDEV
    operator const T*() const { return m_data; }

    HOSTDEV
    T *operator&() { return m_data; }

    HOSTDEV
    const T *operator&() const { return m_data; }

    HOSTDEV
    pixel_proxy<C> operator*() 
        { return pixel_proxy<C>(*this); }

    HOSTDEV
    const_pixel_proxy<C> operator*() const
        { return const_pixel_proxy<C>(*this); }

    HOSTDEV
    dimage_ptr &operator++()
    {
        ++m_data;
        return *this;
    }

    HOSTDEV
    dimage_ptr operator++(int)
    {
        dimage_ptr ret(*this);
        ++*this;
        return ret;
    }

    HOSTDEV
    dimage_ptr &operator--()
    {
        --m_data;
        return *this;
    }

    HOSTDEV
    dimage_ptr operator--(int)
    {
        dimage_ptr ret(*this);
        --*this;
        return ret;
    }

    HOSTDEV
    dimage_ptr &operator+=(int off)
    {
        m_data += off;
        return *this;
    }

    HOSTDEV
    dimage_ptr &operator-=(int off)
    {
        return operator+=(-off);
    }

private:
    template <class U, int D>
    friend class dimage_ptr;

    HOSTDEV
    int channelstride() const { return rowstride()*height(); } 

    int m_width, m_height, m_rowstride;
    texel_type *m_data;
};

template <class T, int C>
dimage_ptr<T,C> dimage<T,C>::operator&()
{
    return dimage_ptr<T,C>(m_data.data(), width(), height(), rowstride());
}

template <class T, int C>
dimage_ptr<const T,C> dimage<T,C>::operator&() const
{
    return dimage_ptr<const T,C>(m_data.data(), width(), height(), rowstride());
}

template <class T, int C>
dimage_ptr<T,1> dimage<T,C>::operator[](int i)
{
    return dimage_ptr<T,1>(m_data.data()+i*channelstride(), 
                           width(), height(), rowstride());
}

template <class T, int C>
dimage_ptr<const T, 1> dimage<T,C>::operator[](int i) const
{
    return dimage_ptr<const T,1>(m_data.data()+i*channelstride(), 
                                 width(), height(), rowstride());
}



#endif
