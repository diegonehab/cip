#ifndef NLFILTER_IMAGE_FRAME_H
#define NLFILTER_IMAGE_FRAME_H

#include <FL/Fl_Gl_Window.H>
#include "nlfilter_gui.h"
#include "dimage.h"

class ImageFrame : public Fl_Gl_Window
{
public:
    ImageFrame(int x=0, int y=0);
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

    void set_grayscale(bool en);

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
