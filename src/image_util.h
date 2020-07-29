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

struct gaussian_blur_plan;

gaussian_blur_plan *gaussian_blur_create_plan(int width, int height,
                                              int rowstride, float sigma);
void free(gaussian_blur_plan *);

void update_plan(gaussian_blur_plan *plan, int width, int height, 
                 int rowstride, float sigma);

template <int C>
void gaussian_blur(gaussian_blur_plan *plan, dimage_ptr<float, C> out, 
                   dimage_ptr<const float,C> in);

template <int C>
void gaussian_blur(gaussian_blur_plan *plan, dimage_ptr<float, C> out, 
                   dimage_ptr<float,C> in)
{
    gaussian_blur(plan, out, dimage_ptr<const float,C>(in));
}

template <int C>
void gaussian_blur(gaussian_blur_plan *plan, dimage_ptr<float, C> out)
{
    gaussian_blur(plan, out, dimage_ptr<const float,C>(out));
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
