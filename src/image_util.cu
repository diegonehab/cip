#include <FL/Fl_PNG_Image.H>
#include <FL/Fl_JPEG_Image.H>
#include <FL/Fl_PNM_Image.H>
#include <FL/filename.H>
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

// compose -----------------------------------------------------

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void compose(float4 *output, 
             const float *r, const float *g, const float *b,
             int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(x >= width || y >= height)
        return;

    int idx = y*rowstride+x;

    output[idx] = make_float4(r[idx],g[idx],b[idx],0);
}

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void compose(uchar4 *output, 
             const float *r, const float *g, const float *b,
             int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(x >= width || y >= height)
        return;

    int idx = y*rowstride+x;

    float3 v = saturate(make_float3(r[idx],g[idx],b[idx]))*255.0f;
    output[idx] = make_uchar4(v.x,v.y,v.z,1);
}

template <class T>
void compose(dvector<T> &out, const dvector<float> in[3], 
             int width, int height, int rowstride)
{
    out.resize(rowstride*height);

    dim3 bdim(BW,BH),
         gdim((width+bdim.x-1)/bdim.x, (height+bdim.y-1)/bdim.y);

    compose<<<gdim, bdim>>>(out, in[0],in[1],in[2], width, height, rowstride);
}

template 
void compose(dvector<float4> &out, const dvector<float> in[3], 
             int width, int height, int rowstride);

template 
void compose(dvector<uchar4> &out, const dvector<float> in[3], 
             int width, int height, int rowstride);


// decompose -----------------------------------------------------

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void decompose(float *r, float *g, float *b, const float4 *input,
                          int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(x >= width || y >= height)
        return;

    int idx = y*rowstride+x;

    float4 v = input[idx];
    r[idx] = v.x;
    g[idx] = v.y;
    b[idx] = v.z;
}

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void decompose(float *r, float *g, float *b, const uchar4 *input,
                          int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(x >= width || y >= height)
        return;

    int idx = y*rowstride+x;

    uchar4 v = input[idx];
    r[idx] = v.x/255.0f;
    g[idx] = v.y/255.0f;
    b[idx] = v.z/255.0f;
}

template <class T>
void decompose(dvector<float> out[3], const dvector<T> &in, 
               int width, int height, int rowstride)
{
    for(int i=0; i<3; ++i)
        out[i].resize(rowstride*height);

    dim3 bdim(BW,BH),
         gdim((width+bdim.x-1)/bdim.x, (height+bdim.y-1)/bdim.y);

    decompose<<<gdim, bdim>>>(out[0],out[1],out[2], in, width, height, rowstride);
}

template
void decompose(dvector<float> out[3], const dvector<float4> &in, 
               int width, int height, int rowstride);

template 
void decompose(dvector<float> out[3], const dvector<uchar4> &in, 
               int width, int height, int rowstride);

// grayscale ------------------------------------------------------------

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void grayscale(float *output, const uchar4 *input,
               int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(x >= width || y >= height)
        return;

    int idx = y*rowstride+x;

    uchar4 in = input[idx];
    output[idx] = in.x/255.0f*0.2126f + in.y/255.0f*0.7152f + in.z/255.0f*0.0722f;
}

__global__
#if USE_LAUNCH_BOUNDS
__launch_bounds__(BW*BH, NB)
#endif
void grayscale(float *output, const float4 *input,
               int width, int height, int rowstride)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    int x = blockIdx.x*BW+tx, y = blockIdx.y*BH+ty;

    if(x >= width || y >= height)
        return;

    int idx = y*rowstride+x;

    float4 in = input[idx];
    output[idx] = in.x*0.2126f + in.y*0.7152f + in.z*0.0722f;
}

template <class T>
void grayscale(dvector<float> &out, const dvector<T> &in, 
               int width, int height, int rowstride)
{
    out.resize(rowstride*height);

    dim3 bdim(BW,BH),
         gdim((width+bdim.x-1)/bdim.x, (height+bdim.y-1)/bdim.y);

    grayscale<<<gdim, bdim>>>(&out, &in, width, height, rowstride);
}


template
void grayscale(dvector<float> &out, const dvector<float4> &in, 
               int width, int height, int rowstride);

template 
void grayscale(dvector<float> &out, const dvector<uchar4> &in, 
               int width, int height, int rowstride);

std::string strupr(const std::string &str)
{
    std::string ret;
    ret.reserve(str.size());

    for(int i=0; i<ret.size(); ++i)
        ret.push_back(toupper(str[i]));
    return ret;
}

void read_image(const std::string &fname, std::vector<uchar4> *data,
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
