#ifndef NLFILTER_IMAGE_UTIL_H
#define NLFILTER_IMAGE_UTIL_H

#include <vector>
#include <string>
#include "dimage.h"

void convert(dimage<float3,1> &out, dimage_ptr<const float,3> in);
void convert(dimage<uchar3,1> &out, dimage_ptr<const float,3> in);
void convert(dimage<float,3> &out, dimage_ptr<const float3,1> in);
void convert(dimage<float,3> &out, dimage_ptr<const uchar3,1> in);

template <class T, int C>
void convert(dimage<T,C> &out, dimage_ptr<const T,C> in)
{
    out = in;
}

void grayscale(dimage<float,1> &out, dimage_ptr<const float3,1> in);
void grayscale(dimage<float,1> &out, dimage_ptr<const uchar3,1> in);

void load_image(const std::string &fname, std::vector<uchar4> *data,
                int *width, int *height);

void save_image(const std::string &fname, const std::vector<uchar4> &data,
                int width, int height);

#endif
