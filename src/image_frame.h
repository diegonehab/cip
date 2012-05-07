#ifndef NLFILTER_IMAGE_FRAME_H
#define NLFILTER_IMAGE_FRAME_H

#include <FL/Fl_Gl_Window.H>
#include "nlfilter_gui.h"

template <class T> class dvector;

class ImageFrame : public Fl_Gl_Window
{
public:
    ImageFrame(const unsigned char *data, int w, int h);
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

    void set_input_image(const unsigned char *data, int w, int h);

    void set_grayscale(bool en);

    // we're returning a pointer to an array of 3 channels
    const dvector<float> *get_input() const;
    dvector<float> *get_output();

    const dvector<float> &get_grayscale_input() const;
    dvector<float> &get_grayscale_output();

    void refresh();
    void swap_buffers();

    int width() const;
    int height() const;
    int rowstride() const;

private:

    virtual void draw();

    struct impl;
    impl *pimpl;
};

#endif
