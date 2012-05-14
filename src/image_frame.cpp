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
        cudaArray *imgarray;

        cudaGraphicsMapResources(1, &cuda_output_resource, 0);
        check_cuda_error("Error mapping output resource");

        try
        {
            cudaGraphicsSubResourceGetMappedArray(&imgarray, 
                cuda_output_resource, 0, 0);
            check_cuda_error("Error getting output cudaArray");

            if(grayscale)
            {
                for(int i=1; i<3; ++i)
                    img_buffer[i] = img_buffer[0];
            }

            assert(!img_buffer.empty());

            temp_buffer.resize(img_buffer.width(), img_buffer.height());

            convert(&temp_buffer, &img_buffer);

            cudaMemcpy2DToArray(imgarray, 0, 0, temp_buffer, 
                                temp_buffer.rowstride()*sizeof(float4),
                                temp_buffer.width()*sizeof(float4), 
                                temp_buffer.height(),
                                cudaMemcpyDeviceToDevice);
            check_cuda_error("Error copying image to array");

            cudaGraphicsUnmapResources(1,&cuda_output_resource, 0);
            check_cuda_error("Error unmapping output resource");

            must_update_texture = false;
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

ImageFrame::ImageFrame(const uchar4 *data, int w, int h)
    : Fl_Gl_Window(0,0,640,480)
{
    pimpl = new impl();

    try
    {
        cudaGLSetGLDevice(0);
        check_cuda_error("Init CUDA-OpenGL interoperability");

        callback(null_cb);

        // this is important, it'll set up GL context properly
        show();

        enable_vsync();

        set_input_image(data, w, h);
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

void ImageFrame::set_input_image(const uchar4 *data, int w, int h)
{
    if(w <= 0 || h  <= 0)
        throw std::runtime_error("Invalid image dimensions");

    make_current();

    if(!pimpl->gl_ok)
        pimpl->initgl();

    dimage<uchar3> d_img;

    d_img.copy_from_host(data, w, h);

    pimpl->img_input.resize(w,h);
    convert(&pimpl->img_input, &d_img);
    grayscale(pimpl->img_input_grayscale, &d_img);

    // to create textures and setup output buffers
    set_grayscale(pimpl->grayscale);

    pimpl->img_buffer.resize(w,h);
    pimpl->img_backbuffer.resize(w,h);

    resize(x(),y(),w,h);

    glViewport(0,0,w,h);
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 
                     pimpl->img_buffer.width(), pimpl->img_buffer.height(), 0,
                     GL_RGBA, GL_FLOAT, NULL);
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
