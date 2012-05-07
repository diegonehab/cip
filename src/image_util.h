#ifndef NLFILTER_IMAGE_UTIL_H
#define NLFILTER_IMAGE_UTIL_H

#include "dvector.h"

template <class T>
void compose(dvector<T> &out, const dvector<float> in[3], 
             int width, int height, int rowstride);


template <class T>
void decompose(dvector<float> out[3], const dvector<T> &in, 
               int width, int height, int rowstride);

template <class T>
void grayscale(dvector<float> &out, const dvector<T> &in,
               int width, int height, int rowstride);

#endif
