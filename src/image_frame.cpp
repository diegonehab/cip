#include <GL/glxew.h>
#include <cuda_gl_interop.h>
#include "image_frame.h"
#include "dvector.h"
#include "recfilter.h"
#include "image_util.h"
#include "threads.h"
#include "vsync.h"

void check_glerror()
{
    std::ostringstream errors;
    GLenum err;
    while((err = glGetError()) != GL_NO_ERROR)
        errors << (const char *)gluErrorString(err) << '\n';

    if(!errors.str().empty())
        throw std::runtime_error("OpenGL: "+errors.str());
};

struct ImageFrame::impl
{
    impl() : gl_ok(false), tex_output(0), cuda_output_resource(NULL)
           , must_update_texture(false)
           , grayscale(false) {}

    dimage<float,3> img_input, img_buffer, img_backbuffer;
    dimage<float,1> img_input_grayscale;

    dimage<float3> temp_buffer;

    bool must_update_texture;

    GLuint tex_output;

    cudaGraphicsResource *cuda_output_resource;

    rod::mutex mtx_buffers;

    bool grayscale;
    bool gl_ok;

    void initgl()/*{{{*/
    {
        assert(!gl_ok);

        glGenTextures(1, &tex_output);
        glBindTexture(GL_TEXTURE_2D, tex_output);
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D, 0);

        glClearColor(0,0,0,1);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        check_glerror();

        gl_ok = true;
    }/*}}}*/

    void deinitgl()/*{{{*/
    {
        if(cuda_output_resource)
        {
            cudaGraphicsUnregisterResource(cuda_output_resource);
            cuda_output_resource = NULL;
        }

        if(tex_output)
        {
            glDeleteTextures(1, &tex_output);
            tex_output = 0;
        }

        gl_ok = false;
    }/*}}}*/

    void update_texture()/*{{{*/
    {
        assert(must_update_texture);

        // let's do this early so that, in case of exceptions, this
        // function won't be called needlessly
        must_update_texture = false;

        cudaArray *imgarray = NULL;

        cudaGraphicsMapResources(1, &cuda_output_resource, 0);
        check_cuda_error("Error mapping output resource");

        try
        {
            cudaGraphicsSubResourceGetMappedArray(&imgarray, 
                cuda_output_resource, 0, 0);
            check_cuda_error("Error getting output cudaArray");

            assert(imgarray != NULL);

            if(grayscale)
            {
                cudaMemcpy2DToArray(imgarray, 0, 0, img_buffer[0], 
                                    img_buffer[0].rowstride()*sizeof(float),
                                    img_buffer[0].width()*sizeof(float), 
                                    img_buffer[0].height(),
                                    cudaMemcpyDeviceToDevice);
                check_cuda_error("Error copying image to array");
            }
            else
            {
                assert(!img_buffer.empty());

                temp_buffer.resize(img_buffer.width(), img_buffer.height());

                convert(&temp_buffer, &img_buffer);

                cudaMemcpy2DToArray(imgarray, 0, 0, temp_buffer, 
                                    temp_buffer.rowstride()*sizeof(float4),
                                    temp_buffer.width()*sizeof(float4), 
                                    temp_buffer.height(),
                                    cudaMemcpyDeviceToDevice);
                check_cuda_error("Error copying image to array");
            }

            cudaGraphicsUnmapResources(1,&cuda_output_resource, 0);
            check_cuda_error("Error unmapping output resource");
        }
        catch(...)
        {
            cudaGraphicsUnmapResources(1,&cuda_output_resource, 0);
            check_cuda_error("Error unmapping output resource");
            throw;
        }
    }/*}}}*/
};

void null_cb(Fl_Widget *, void*) {}

ImageFrame::ImageFrame(int x, int y)
    : Fl_Gl_Window(x,y,640,480)
{
    pimpl = new impl();

    try
    {
        // disables window closing
        callback(null_cb);

        resizable(this);

        show();
        make_current();

        static bool device_init = false;

        if(!device_init)
        {
            cudaGLSetGLDevice(0);
            check_cuda_error("Init CUDA-OpenGL interoperability");
            device_init = true;
        }
//        enable_vsync();
    }
    catch(...)
    {
        delete pimpl;
        throw;
    }
}

ImageFrame::~ImageFrame()
{
    try
    {
        make_current();
        pimpl->deinitgl();
    }
    catch(std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
    catch(...)
    {
        std::cerr << "Unexpected error" << std::endl;
    }

    delete pimpl;
}

void ImageFrame::set_input_image(dimage_ptr<const float3> img)
{
    if(img.width() <= 0 || img.height()  <= 0)
        throw std::runtime_error("Invalid image dimensions");

    make_current();

    if(!pimpl->gl_ok)
        pimpl->initgl();

    pimpl->img_input.resize(img.width(),img.height());
    convert(&pimpl->img_input, img);

    pimpl->img_input_grayscale.resize(img.width(), img.height());
    grayscale(&pimpl->img_input_grayscale, img);

    // to create textures and setup output buffers
    set_grayscale(pimpl->grayscale);

    pimpl->img_buffer.resize(img.width(),img.height());
    pimpl->img_backbuffer.resize(img.width(),img.height());

    resize(x(),y(),img.width(),img.height());
}

void ImageFrame::set_input_image(dimage_ptr<const float,3> img)
{
    if(img.width() <= 0 || img.height()  <= 0)
        throw std::runtime_error("Invalid image dimensions");

    make_current();

    if(!pimpl->gl_ok)
        pimpl->initgl();

    pimpl->img_input = img;

    pimpl->img_input_grayscale.resize(img.width(), img.height());
    grayscale(&pimpl->img_input_grayscale, img);

    // to create textures and setup output buffers
    set_grayscale(pimpl->grayscale);

    pimpl->img_buffer.resize(img.width(),img.height());
    pimpl->img_backbuffer.resize(img.width(),img.height());

    resize(x(),y(),img.width(),img.height());

    glViewport(0,0,img.width(),img.height());
    check_glerror();
}

void ImageFrame::set_grayscale(bool en)
{
    pimpl->grayscale = en;

    if(pimpl->grayscale)
        pimpl->img_buffer[0] = pimpl->img_input_grayscale;
    else
        pimpl->img_buffer = pimpl->img_input;

    if(pimpl->cuda_output_resource)
    {
        cudaGraphicsUnregisterResource(pimpl->cuda_output_resource);
        check_cuda_error("Error unregistering resource");
        pimpl->cuda_output_resource = NULL;
    }

    glBindTexture(GL_TEXTURE_2D, pimpl->tex_output);

    try
    {
        if(en)
        {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB,
                         pimpl->img_buffer.width(),pimpl->img_buffer.height(),0,
                         GL_LUMINANCE, GL_FLOAT, NULL);
        }
        else
        {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 
                         pimpl->img_buffer.width(),pimpl->img_buffer.height(),0,
                         GL_RGBA, GL_FLOAT, NULL);
        }
        check_glerror();

        glBindTexture(GL_TEXTURE_2D, 0);
    }
    catch(...)
    {
        glBindTexture(GL_TEXTURE_2D, 0);
        throw;
    }

    cudaGraphicsGLRegisterImage(&pimpl->cuda_output_resource, pimpl->tex_output,
                    GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    check_cuda_error("Error registering GL image");

    refresh();
}

void ImageFrame::refresh()
{
    pimpl->must_update_texture = true;
    redraw();
}

void ImageFrame::swap_buffers()
{
    using std::swap;

    rod::unique_lock lk(pimpl->mtx_buffers);

    if(!pimpl->img_backbuffer.empty())
        swap(pimpl->img_backbuffer, pimpl->img_buffer);

    lk.unlock();

    refresh();
}

void ImageFrame::reset()
{
    if(pimpl->grayscale)
        pimpl->img_backbuffer[0] = pimpl->img_input_grayscale;
    else
        pimpl->img_backbuffer = pimpl->img_input;

    swap_buffers();
}

void ImageFrame::resize (int x, int y, int w, int h)
{
    Fl_Gl_Window::resize(x,y,w,h);

    glViewport(0,0,w,h);
    check_glerror();
}


void ImageFrame::draw()
{
    try
    {
        if(!pimpl->gl_ok)
            pimpl->initgl();

        if(pimpl->must_update_texture)
            pimpl->update_texture();

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, pimpl->tex_output);

        float pu = 0.5/w(),
              pv = 0.5/h();

        glClear(GL_COLOR_BUFFER_BIT);

        glBegin(GL_QUADS);
            glTexCoord2f(pu, 1-pv);
            glVertex2f(-1,-1);

            glTexCoord2f(1-pu, 1-pv);
            glVertex2f(1,-1);

            glTexCoord2f(1-pu, pv);
            glVertex2f(1,1);

            glTexCoord2f(pu, pv);
            glVertex2f(-1,1);
        glEnd();

        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);
    }
    catch(std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
}

dimage_ptr<float,3> ImageFrame::get_output()
{
    return &pimpl->img_backbuffer;
}

dimage_ptr<const float,3> ImageFrame::get_input() const
{
    return &pimpl->img_input;
}

dimage_ptr<float> ImageFrame::get_grayscale_output()
{
    return pimpl->img_backbuffer[0];
}
dimage_ptr<const float,1> ImageFrame::get_grayscale_input() const
{
    return &pimpl->img_input_grayscale;
}

ImageFrame::OutputBufferLocker::OutputBufferLocker(ImageFrame &imgframe)
    : m_locked(false)
    , m_imgframe(imgframe)
{
    m_imgframe.pimpl->mtx_buffers.lock();
    m_locked = true;
}

ImageFrame::OutputBufferLocker::~OutputBufferLocker()
{
    if(m_locked)
        m_imgframe.pimpl->mtx_buffers.unlock();
}

void ImageFrame::OutputBufferLocker::unlock()
{
    m_imgframe.pimpl->mtx_buffers.unlock();
}
