#ifndef NLFILTER_IMAGE_UTIL_H
#define NLFILTER_IMAGE_UTIL_H

#include <vector>
#include <string>
#include "dimage.h"

void convert(dimage<float4,1> &out, dimage_ptr<const float,3> in);
void convert(dimage<uchar4,1> &out, dimage_ptr<const float,3> in);
void convert(dimage<float,3> &out, dimage_ptr<const float4,1> in);
void convert(dimage<float,3> &out, dimage_ptr<const uchar4,1> in);

template <class T, int C>
void convert(dimage<T,C> &out, dimage_ptr<const T,C> in)
{
    out = in;
}

void grayscale(dimage<float,1> &out, dimage_ptr<const float4,1> in);
void grayscale(dimage<float,1> &out, dimage_ptr<const uchar4,1> in);

void load_image(const std::string &fname, std::vector<uchar4> *data,
                int *width, int *height);

void save_image(const std::string &fname, const std::vector<uchar4> &data,
                int width, int height);

#endif
