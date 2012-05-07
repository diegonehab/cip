#ifndef UTIL_IMAGE_H
#define UTIL_IMAGE_H

enum BorderType
{
    CLAMP_TO_ZERO,
    CLAMP_TO_EDGE,
    REPEAT,
    REFLECT
};

float adjcoord(float t, BorderType border_type);

int calcidx(int x, int y, int w, int h, BorderType border_type);

float getpix(const float *data, int x, int y, int w, int h, 
             BorderType border_type);

float *extend_image(const float *img, int w, int h, 
                    int border_top, int border_left,
                    int border_bottom, int border_right,
                    BorderType border_type);

void calc_borders(int *left, int *top, int *right, int *bottom,
                  int w, int h, int border=0);

void crop_image(float *cimg, const float *img, int w, int h, 
                int border_top, int border_left,
                int border_bottom, int border_right);


#endif
