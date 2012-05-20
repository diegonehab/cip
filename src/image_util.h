#ifndef NLFILTER_IMAGE_UTIL_H
#define NLFILTER_IMAGE_UTIL_H

#include <vector>
#include <string>
#include "dimage.h"
#include "util.h"

// convert ----------------------------

template <class T, int C, class U, int D>
void convert(dimage_ptr<T,C> out, dimage_ptr<const U,D> in);

template <class T, int C, class U, int D>
void convert(dimage_ptr<T,C> out, dimage_ptr<U,D> in)
{
    convert(out, dimage_ptr<const U,D>(in));
}

template <class T, int C>
void convert(dimage_ptr<T,C> out, dimage_ptr<const T,C> in)
{
    out = in;
}

// grayscale ----------------------------

template <class T, int C>
void grayscale(dimage_ptr<float> out, dimage_ptr<const T,C> in);

template <class T, int C>
void grayscale(dimage_ptr<float> out, dimage_ptr<T,C> in)
{
    grayscale(out, dimage_ptr<const T,C>(in));
}

// convolve ----------------------------

template <class T, int C, class U, int D, int R>
void convolve(dimage_ptr<T,C> out, dimage_ptr<const U,D> in,
              const array<float,R> &kernel, int scale=1);

template <class T, int C, class U, int D, int R>
void convolve(dimage_ptr<T,C> out, dimage_ptr<U,D> in,
              const array<float,R> &kernel, int scale=1)
{
    convolve(out, dimage_ptr<const U,D>(in), kernel, scale);
}

// lrgb2srgb ----------------------------

template <class T, int C, class U, int D>
void lrgb2srgb(dimage_ptr<T,C> out, dimage_ptr<const U,D> in);

template <class T, int C, class U, int D>
void lrgb2srgb(dimage_ptr<T,C> out, dimage_ptr<U,D> in)
{
    lrgb2srgb(out, dimage_ptr<const U,D>(in));
}

// luminance -----------------------------
//
template <class T, int C>
void luminance(dimage_ptr<float> out, dimage_ptr<const T,C> in);

template <class T, int C>
void luminance(dimage_ptr<float> out, dimage_ptr<T,C> in)
{
    luminance(out, dimage_ptr<const T,C>(in));
}

// gaussian blur ------------------------------------------

template <int C>
void gaussian_blur(dimage_ptr<float, C> out, dimage_ptr<const float,C> in,
                   float sigma);

template <int C>
void gaussian_blur(dimage_ptr<float, C> out, dimage_ptr<float,C> in,
                   float sigma)
{
    gaussian_blur(out, dimage_ptr<const float,C>(in), sigma);
}

template <int C>
void gaussian_blur(dimage_ptr<float, C> out, float sigma)
{
    gaussian_blur( out, dimage_ptr<const float,C>(out), sigma);
}

// I/O ----------------------------

void load_image(const std::string &fname, std::vector<uchar4> *data,
                int *width, int *height);

void save_image(const std::string &fname, const std::vector<uchar4> &data,
                int width, int height);

void save_image(const std::string &fname, const std::vector<unsigned char> &data,
                 int width, int height);

template <class T, int C>
void save_image(const std::string &fname, dimage_ptr<const T,C> img);

template <class T, int C>
void save_image(const std::string &fname, dimage_ptr<T,C> img)
{
    save_image(fname, dimage_ptr<const T,C>(img));
}

#endif
