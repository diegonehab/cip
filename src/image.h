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
