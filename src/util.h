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

#endif
