#ifndef SYMBOL_H
#define SYMBOL_H

#include <string>
#include <vector>
#include "error.h"

template <class T>
void copy_to_symbol(const std::string &name, const T &value)
{
    size_t size_storage;
    cudaGetSymbolSize(&size_storage, name.c_str());
    check_cuda_error("Invalid symbol '"+name+"'");

    if(sizeof(T) > size_storage)
        throw std::runtime_error("'"+name+"'"+" storage overflow");

    cudaMemcpyToSymbol(name.c_str(), &value, sizeof(T), 0,
                       cudaMemcpyHostToDevice);
    check_cuda_error("Error copying '"+name+"' buffer to device");
}

inline void copy_to_symbol(const std::string &name, unsigned long value)
{
    copy_to_symbol(name, (unsigned int)value);
}

inline void copy_to_symbol(const std::string &name, long value)
{
    copy_to_symbol(name, (int)value);
}

template <class T>
void copy_to_symbol(const std::string &name, const std::string &size_name,
                    const std::vector<T> &items)
{
    size_t size_storage;
    cudaGetSymbolSize(&size_storage, name.c_str());
    check_cuda_error("Invalid symbol '"+name+"'");

    size_t size = items.size()*sizeof(T);

    if(size > size_storage)
        throw std::runtime_error("'"+name+"'"+" storage overflow");

    cudaMemcpyToSymbol(name.c_str(),&items[0], size, 0,
                       cudaMemcpyHostToDevice);
    check_cuda_error("Error copying '"+name+"' buffer to device");

    if(!size_name.empty())
        copy_to_symbol(size_name.c_str(), items.size());
}

template <class T>
void copy_to_symbol(const std::string &name, const std::vector<T> &items)
{
    copy_to_symbol(name, "", items);
}

template <class T>
T copy_from_symbol(const std::string &name)
{
    size_t size_storage;
    cudaGetSymbolSize(&size_storage, name.c_str());
    check_cuda_error("Invalid symbol '"+name+"'");

    if(sizeof(T) > size_storage)
        throw std::runtime_error("'"+name+"'"+" storage overflow");

    T value;
    cudaMemcpyFromSymbol(&value, name.c_str(), sizeof(T), 0,
                       cudaMemcpyDeviceToHost);
    check_cuda_error("Error copying '"+name+"' buffer to device");

    return value;
}


#endif
