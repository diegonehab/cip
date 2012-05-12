#ifndef GPUFILTER_UTIL_H
#define GPUFILTER_UTIL_H

inline std::string strupr(const std::string &str)
{
    std::string ret;
    ret.reserve(str.size());

    for(int i=0; i<ret.size(); ++i)
        ret.push_back(toupper(str[i]));
    return ret;
}

template <bool EN, class T=void>
struct enable_if
{
    typedef T type;
};

template <class T>
struct enable_if<false,T>
{
};

template <class FROM, class TO>
struct is_convertible
{
private:
    struct yes {};
    struct no {yes m[2]; };

    static yes can_convert(TO *) {}
    static no can_convert(const void *) {}

public:
    static const bool value 
        = sizeof(can_convert((FROM *)NULL)) == sizeof(yes);
};

template <class T>
struct remove_const
{
    typedef T type;
};

template <class T>
struct remove_const<const T>
{
    typedef T type;
};

template <class T, int N>
class array
{
    T m_data[N];
public:
    __host__ __device__
    const T &operator[](int i) const { return m_data[i]; }

    __host__ __device__
    T &operator[](int i) { return m_data[i]; }
};

template <class T, int N>
class array<const T *, N>
{
    const T *m_data[N];
public:
    __host__ __device__
    array()
    {
    }

    __host__ __device__
    array(const array &that)
    {
        for(int i=0; i<N; ++i)
            m_data[i] = that[i];
    }

    __host__ __device__
    array(const array<T*,N> &that)
    {
        for(int i=0; i<N; ++i)
            m_data[i] = that[i];
    }

    __host__ __device__
    const T *&operator[](int i) { return m_data[i]; }

    __host__ __device__
    const T *const&operator[](int i) const { return m_data[i]; }
};

template <class FROM, class TO>
struct copy_const
{
    typedef TO type;
};

template <class FROM, class TO>
struct copy_const<const FROM, TO>
{
    typedef const TO type;
};


#endif
