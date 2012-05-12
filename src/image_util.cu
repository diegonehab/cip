#include <FL/Fl_PNG_Image.H>
#include <FL/Fl_JPEG_Image.H>
#include <FL/Fl_PNM_Image.H>
#include <FL/filename.H>
#include <cutil.h>
#include "math_util.h"
#include "image_util.h"

#define USE_LAUNCH_BOUNDS 1

const int BW = 32, // cuda block width
          BH = 6, // cuda block height
#if CUDA_SM >= 20
          NB = 8;
#else
          NB = 5;
#endif

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void convert(dimage_ptr<float3,1> out, dimage_ptr<const float,3> in)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(!in.is_inside(x,y))
        return;

    int idx = in.offset_at(x,y);
    in += idx;
    out += idx;

    *out = *in;
}

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void convert(dimage_ptr<uchar3,1> out, dimage_ptr<const float,3> in)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(!in.is_inside(x,y))
        return;

    int idx = in.offset_at(x,y);
    in += idx;
    out += idx;

    float3 v = saturate(*in)*255.0f;
    *out = make_uchar3(v.x,v.y,v.z);
}

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void convert(dimage_ptr<float,3> out, dimage_ptr<const float3,1> in)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(!in.is_inside(x,y))
        return;

    int idx = in.offset_at(x,y);
    in += idx;
    out += idx;

    *out = *in;
}

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void convert(dimage_ptr<float,3> out, dimage_ptr<const uchar3,1> in)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(!in.is_inside(x,y))
        return;

    int idx = in.offset_at(x,y);
    in += idx;
    out += idx;

    uchar3 v = *in;
    *out = make_float3(v.x/255.0,v.y/255.0,v.z/255.0);
}

template <class T, int CT, class U, int CU>
void call_convert(dimage<T,CT> &out, dimage_ptr<const U,CU> in)
{
    out.resize(in.width(), in.height(), in.rowstride());

    dim3 bdim(BW,BH),
         gdim((in.width()+bdim.x-1)/bdim.x, (in.height()+bdim.y-1)/bdim.y);

    convert<<<gdim, bdim>>>(&out, in);
}

void convert(dimage<float3,1> &out, dimage_ptr<const float,3> in)
{
    call_convert(out, in);
}
void convert(dimage<uchar3,1> &out, dimage_ptr<const float,3> in)
{
    call_convert(out, in);
}

void convert(dimage<float,3> &out, dimage_ptr<const float3,1> in)
{
    call_convert(out, in);
}
void convert(dimage<float,3> &out, dimage_ptr<const uchar3,1> in)
{
    call_convert(out, in);
}

// grayscale ------------------------------------------------------------

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void grayscale(dimage_ptr<float,1> out, dimage_ptr<const uchar3,1> in)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(!in.is_inside(x,y))
        return;

    int idx = in.offset_at(x,y);
    in += idx;
    out += idx;

    uchar3 p = *in;
    *out = p.x/255.0f*0.2126f + p.y/255.0f*0.7152f + p.z/255.0f*0.0722f;
}

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void grayscale(dimage_ptr<float,1> out, dimage_ptr<const float3,1> in)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(!in.is_inside(x,y))
        return;

    int idx = in.offset_at(x,y);

    float3 p = *in;
    *out = p.x*0.2126f + p.y*0.7152f + p.z*0.0722f;
}

template <class T>
void call_grayscale(dimage<float,1> &out, dimage_ptr<const T,1> in)
{
    out.resize(in.width(), in.height(), in.rowstride());

    dim3 bdim(BW,BH),
         gdim((in.width()+bdim.x-1)/bdim.x, (in.height()+bdim.y-1)/bdim.y);

    grayscale<<<gdim, bdim>>>(&out, in);
}


void grayscale(dimage<float,1> &out, dimage_ptr<const float3,1> in)
{
    call_grayscale(out, in);
}

void grayscale(dimage<float,1> &out, dimage_ptr<const uchar3,1> in)
{
    call_grayscale(out, in);
}

void load_image(const std::string &fname, std::vector<uchar4> *data,
                int *width, int *height)
{
    // Reads 'fname' into an Fl_Image
    Fl_Image *img;

    std::string FNAME = fname;
    strupr(const_cast<char *>(FNAME.data()));

    if(fl_filename_match(FNAME.c_str(),"*.PNG"))
        img = new Fl_PNG_Image(fname.c_str());
    else if(fl_filename_match(FNAME.c_str(),"*.JPG"))
        img = new Fl_JPEG_Image(fname.c_str());
    else if(fl_filename_match(FNAME.c_str(),"*.{PNM,PBM,PGM,PPM}"))
        img = new Fl_PNM_Image(fname.c_str());
    else
        throw std::runtime_error("Image type not supported");

    if(img->w()==0 || img->h()==0)
        throw std::runtime_error("Error loading image");

    if(width)
        *width = img->w();

    if(height)
        *height = img->h();

    // creates an RGBA array out of Fl_Image internal image representation
    if(data != NULL)
    {
        data->clear();
        data->reserve(img->w()*img->h());

        int irow = img->w()*img->d()+img->ld();
        unsigned char *currow = (unsigned char *)img->data()[0];

        // grayscale?
        if(img->d() < 3)
        {
            for(int i=0; i<img->h(); ++i, currow += irow)
            {
                for(int j=0; j<img->w(); ++j)
                {
                    int p = j*img->d();

                    uchar4 outp;
                    outp.x = outp.y = outp.z = currow[p];

                    // has alpha channel?
                    if(img->d() > 1)
                        outp.w = currow[p+1];
                    else
                        outp.w = 255;

                    data->push_back(outp);
                }
            }
        }
        // full RGB
        else
        {
            for(int i=0; i<img->h(); ++i, currow += irow)
            {
                for(int j=0; j<img->w(); ++j)
                {
                    int p = j*img->d();

                    uchar4 outp;
                    outp.x = currow[p];
                    outp.y = currow[p+1];
                    outp.z = currow[p+2];

                    // has alpha channel?
                    if(img->d() > 3)
                        outp.w = currow[p+3];
                    else
                        outp.w = 255;

                    data->push_back(outp);
                }
            }
        }
    }
}

void save_image(const std::string &fname, const std::vector<uchar4> &data,
                 int width, int height)
{
    if(fl_filename_match(strupr(fname).c_str(),"*.PPM"))
        throw std::runtime_error("We only support PPM output image format");

    if(!cutSavePPM4ub(fname.c_str(), (unsigned char *)&data[0], width, height))
        throw std::runtime_error("Error saving output image");
}

