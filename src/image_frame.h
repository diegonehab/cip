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

#ifndef NLFILTER_IMAGE_FRAME_H
#define NLFILTER_IMAGE_FRAME_H

#include <FL/Fl_Gl_Window.H>
#include "nlfilter_gui.h"
#include "dimage.h"

class ImageFrame : public Fl_Gl_Window
{
public:
    ImageFrame(int x=0, int y=0, int w=64, int h=64);
    ~ImageFrame();

    class OutputBufferLocker
    {
    public:
        OutputBufferLocker(ImageFrame &imgframe);
        ~OutputBufferLocker();

        void unlock();

    private:
        ImageFrame &m_imgframe;
        bool m_locked;
    };

    void set_input_image(dimage_ptr<const float3> img);
    void set_input_image(dimage_ptr<const float,3> img);

    void set_blank_input_image(int w, int h);

    void set_grayscale(bool en);

    void reset();

    dimage_ptr<const float,3> get_input() const;
    dimage_ptr<float,3> get_output();

    dimage_ptr<const float,1> get_grayscale_input() const;
    dimage_ptr<float,1> get_grayscale_output();

    void refresh();
    void swap_buffers();
private:

    virtual void draw();

    struct impl;
    impl *pimpl;
};

#endif
