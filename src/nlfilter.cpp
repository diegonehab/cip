#include <FL/fl_ask.H> // for fl_alert
#include <FL/Fl_File_Chooser.H>
#include <FL/Fl_PNG_Image.H>
#include <FL/Fl_JPEG_Image.H>
#include <FL/Fl_PNM_Image.H>
#include <FL/Fl_Color_Chooser.H>
#include <vector>
#include <cuda_gl_interop.h>
#include "image_util.h"
#include "recfilter.h"
#include "timer.h"
#include "filter.h"
#include "config.h"
#include "nlfilter_gui.h"
#include "image_frame.h"
#include "threads.h"

class MainFrame : public MainFrameUI
{
public:
    MainFrame();
    ~MainFrame();

    void open(const char *fname);

    void on_file_open();
    void on_file_save();
    void on_file_save_as();
    void on_file_exit();

    void on_choose_effect(effect_type effect);
    void on_apply_effect();
    void on_undo_effect();
    void on_change_grayscale(bool gs);

    static void on_param_changed(Fl_Widget *,MainFrame *frame)
    {
        // update the image to reflect the parameter change
        frame->update_image();
    }

    static void on_color_change(Fl_Widget *w, MainFrame *frame)
    {
        Fl_Button *btn = dynamic_cast<Fl_Button *>(w);
        assert(btn);

        unsigned char r, g, b;
        Fl::get_color(btn->color(), r, g, b);

        if(fl_color_chooser(btn->label(), r, g, b))
        {
            btn->color(fl_rgb_color(r,g,b));
            frame->update_image();
        }
    }

    virtual void show()
    {
        MainFrameUI::show();
        if(m_image_frame)
            m_image_frame->show();
    }
    virtual void hide()
    {
        if(m_image_frame)
            m_image_frame->hide();
        MainFrameUI::hide();
    }

private:
    void update_image();

    static void *render_thread(MainFrame *frame);

    rod::thread m_render_thread;
    rod::condition_variable m_wakeup; // wakes up the render thread
    rod::mutex m_mtx_render_data;
    bool m_terminate_thread; // tells the render thread to finish its business
    bool m_has_new_render_job;

    Fl_Group *m_param_panel; // current effect parameter panel
    ImageFrame *m_image_frame; 
};

MainFrame::MainFrame()
    : MainFrameUI(753,319,595,125, "NLFilter")
    , m_render_thread((void(*)(void*))&MainFrame::render_thread, this, false)
    , m_param_panel(NULL)
    , m_image_frame(NULL)
    , m_terminate_thread(false)
    , m_has_new_render_job(false)
{
    m_effects->add("Identity",0,NULL,(void*)EFFECT_IDENTITY);
    m_effects->add("Posterize",0,NULL,(void*)EFFECT_POSTERIZE);
    m_effects->add("Scale",0,NULL,(void*)EFFECT_SCALE);
    m_effects->add("Bias",0,NULL,(void*)EFFECT_BIAS);
    m_effects->add("Root",0,NULL,(void*)EFFECT_ROOT);
    m_effects->add("Threshold",0,NULL,(void*)EFFECT_THRESHOLD);
    m_effects->add("Replacement",0,NULL,(void*)EFFECT_REPLACEMENT);
    m_effects->value(0);

    // kicks off the render thread
    m_render_thread.start();
}

MainFrame::~MainFrame()
{
    // signal the render thread to terminate
    rod::unique_lock lk(m_mtx_render_data);
    m_terminate_thread = true;
    m_wakeup.signal();
    lk.unlock();

    // just wait until it finishes
    m_render_thread.join();
}

// defined on timer.cpp
std::string unit_value(double v, double base);

void *MainFrame::render_thread(MainFrame *frame)
{
    while(!frame->m_terminate_thread)
    {
        rod::unique_lock lk(frame->m_mtx_render_data);

        ImageFrame *imgframe = frame->m_image_frame;

        // no render job? sleep
        if(imgframe == NULL || !frame->m_has_new_render_job)
            frame->m_wakeup.wait(lk);

        // maybe it was set during wait
        imgframe = frame->m_image_frame;

        if(!frame->m_terminate_thread && imgframe != NULL &&
           frame->m_has_new_render_job)
        {
            // fill the operation struct along with its parameters based
            // on what is set by the user
            filter_operation op;

            op.type = (effect_type)(ptrdiff_t)frame->m_effects->mvalue()->user_data_;

            if(const ParamPosterizeUI *panel = dynamic_cast<const ParamPosterizeUI *>(frame->m_param_panel))
                op.levels = std::max<int>(2,panel->levels->value());
            else if(const ParamScaleUI *panel = dynamic_cast<const ParamScaleUI *>(frame->m_param_panel))
                op.scale = panel->scale->value();
            else if(const ParamBiasUI *panel = dynamic_cast<const ParamBiasUI *>(frame->m_param_panel))
                op.bias = panel->bias->value();
            else if(const ParamRootUI *panel = dynamic_cast<const ParamRootUI *>(frame->m_param_panel))
                op.degree = panel->degree->value();
            else if(const ParamThresholdUI *panel = dynamic_cast<const ParamThresholdUI *>(frame->m_param_panel))
                op.threshold = panel->threshold->value();
            else if(const ParamReplacementUI *panel = dynamic_cast<const ParamReplacementUI *>(frame->m_param_panel))
            {
                unsigned char r,g,b;
                Fl::get_color(panel->old_color->color(), r, g, b);
                op.old_color = make_float3(r/255.0f, g/255.0f, b/255.0f);

                Fl::get_color(panel->new_color->color(), r, g, b);
                op.new_color = make_float3(r/255.0f, g/255.0f, b/255.0f);

                op.tau.x = panel->tau_red->value();
                op.tau.y = panel->tau_green->value();
                op.tau.z = panel->tau_blue->value();
            }

            frame->m_has_new_render_job = false;

            lk.unlock();

            // lock buffers since we'll write on them
            ImageFrame::OutputBufferLocker lkbuffers(*imgframe);

            cpu_timer timer;

            // just process one (grayscale) channel?
            if(frame->m_grayscale->value())
            {
                recursive_filter_5(imgframe->get_grayscale_output(),
                                   imgframe->get_grayscale_input());

                filter(imgframe->get_grayscale_output(), imgframe->width(), 
                       imgframe->height(), imgframe->rowstride(), op);

                recursive_filter_5(imgframe->get_grayscale_output());
            }
            else
            {
                // convolve with a bpsline3^-1 to make a cardinal post-filter
                for(int i=0; i<3; ++i)
                {
                    recursive_filter_5(imgframe->get_output()[i],
                                       imgframe->get_input()[i]);
                }

                // do actual filtering
#if USE_SM20
                filter(imgframe->get_output(), imgframe->width(), 
                       imgframe->height(), imgframe->rowstride(), op);
#else
                for(int i=0; i<3; ++i)
                {
                    filter(imgframe->get_output()[i], imgframe->width(), 
                           imgframe->height(), imgframe->rowstride(), op);
                }
#endif

                // convolve with a bpsline3^-1 to make a cardinal pre-filter
                for(int i=0; i<3; ++i)
                    recursive_filter_5(imgframe->get_output()[i]);
            }

            timer.stop();

            // done working with buffers, release the lock
            lkbuffers.unlock();

            // avoids a deadlock in Fl::lock() because usually we're
            // destroying the main window if terminate_thread is true
            // at this point
            if(frame->m_terminate_thread)
                break;

            // tell the image window to update its content and update
            // performance counters
            Fl::lock();

            frame->m_status_fps->value(1.0/timer.elapsed());
            float rate = imgframe->width()*imgframe->height() / timer.elapsed();
            std::string srate = unit_value(rate, 1000), unit;
            {
                std::istringstream ss(srate);
                ss >> rate >> unit;
                unit += "P/s ";
            }
            frame->m_status_rate->copy_label(unit.c_str());
            frame->m_status_rate->value(rate);

            imgframe->swap_buffers();
            Fl::unlock();
            Fl::awake((void *)NULL); // awake message loop processing
        }
    }
}

void MainFrame::on_change_grayscale(bool gs)
{
    if(m_image_frame)
        m_image_frame->set_grayscale(gs);
    update_image();
}

void strupr(char *str)
{
    while(*str)
    {
        *str = toupper(*str);
        ++str;
    }
}

void MainFrame::open(const char *fname)
{
    // Reads 'fname' into an Fl_Image
    Fl_Image *img;

    std::string FNAME = fname;
    strupr(const_cast<char *>(FNAME.data()));

    if(fl_filename_match(FNAME.c_str(),"*.PNG"))
        img = new Fl_PNG_Image(fname);
    else if(fl_filename_match(FNAME.c_str(),"*.JPG"))
        img = new Fl_JPEG_Image(fname);
    else if(fl_filename_match(FNAME.c_str(),"*.{PNM,PBM,PGM,PPM}"))
        img = new Fl_PNM_Image(fname);
    else
        throw std::runtime_error("Image type not supported");

    if(img->w()==0 || img->h()==0)
        throw std::runtime_error("Error loading image");

    // creates an RGBA array out of Fl_Image internal image representation
    std::vector<unsigned char> imgdata;
    imgdata.reserve(img->w()*img->h()*4);

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

                imgdata.push_back(currow[p]);
                imgdata.push_back(currow[p]);
                imgdata.push_back(currow[p]);

                // has alpha channel?
                if(img->d() > 1)
                    imgdata.push_back(currow[p+1]);
                else
                    imgdata.push_back(255);
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

                imgdata.push_back(currow[p]);
                imgdata.push_back(currow[p+1]);
                imgdata.push_back(currow[p+2]);

                // has alpha channel?
                if(img->d() > 3)
                    imgdata.push_back(currow[p+3]);
                else
                    imgdata.push_back(255);
            }
        }
    }

    // create the image frame with the image data
    if(!m_image_frame)
    {
        m_image_frame = new ImageFrame(&imgdata[0], img->w(), img->h());
        m_image_frame->show();
    }
    else
        m_image_frame->set_input_image(&imgdata[0], img->w(), img->h());

    m_image_frame->copy_label(fname);

    // setup recursive filters and other stuff

    // calculate cubic b-spline weights
    Vector<float,1+1> weights;
    {
        float a = 2.f-std::sqrt(3.0f);

        weights[0] = 1+a;
        weights[1] = a;
    }

    init_blue_noise();

    recursive_filter_5_setup(m_image_frame->width(), 
                             m_image_frame->height(), 
                             m_image_frame->rowstride(), 
                             weights, CLAMP_TO_EDGE, 1);
}

void MainFrame::on_file_open()
{
    Fl_File_Chooser dlg(".","Images (*.{jpg,png,ppm,pgm,pnm,pbm})\t"
                            "All (*)", Fl_File_Chooser::SINGLE,
                            "Open Image");

    dlg.show();

    while(dlg.shown())
        Fl::wait();

    if(dlg.count() == 0)
        return;

    open(dlg.value(1));
}

void MainFrame::on_file_save()
{
}

void MainFrame::on_file_save_as()
{
}

void MainFrame::on_file_exit()
{
}

void MainFrame::on_choose_effect(effect_type effect)
{
    Fl_Group *panel = NULL;

    int px = m_param_group->x(),
        py = m_param_group->y(),
        pw = m_param_group->w(),
        ph = m_param_group->h();

    m_param_group->begin();

    // function to be called when a parameter changes 
    // (it should update the image)
    Fl_Callback *on_param_changed = (Fl_Callback *)&MainFrame::on_param_changed;
    Fl_Callback *on_color_change = (Fl_Callback *)&MainFrame::on_color_change;

    // creates the panel associated with the selected effect
    switch(effect)
    {
    case EFFECT_IDENTITY:
        panel = NULL;
        break;
    case EFFECT_POSTERIZE:
        {
            ParamPosterizeUI *_panel = new ParamPosterizeUI(0,0,pw,ph);
            _panel->levels->callback(on_param_changed, this);
            panel = _panel;
        }
        break;
    case EFFECT_SCALE:
        {
            ParamScaleUI *_panel = new ParamScaleUI(0,0,pw,ph);
            _panel->scale->callback(on_param_changed, this);
            panel = _panel;
        }
        break;
    case EFFECT_BIAS:
        {
            ParamBiasUI *_panel = new ParamBiasUI(0,0,pw,ph);
            _panel->bias->callback(on_param_changed, this);
            panel = _panel;
        }
        break;
    case EFFECT_ROOT:
        {
            ParamRootUI *_panel = new ParamRootUI(0,0,pw,ph);
            _panel->degree->callback(on_param_changed, this);
            panel = _panel;
        }
        break;
    case EFFECT_THRESHOLD:
        {
            ParamThresholdUI *_panel = new ParamThresholdUI(0,0,pw,ph);
            _panel->threshold->callback(on_param_changed, this);
            panel = _panel;
        }
    case EFFECT_REPLACEMENT:
        {
            ParamReplacementUI *_panel = new ParamReplacementUI(0,0,pw,ph);
            _panel->tau_red->callback(on_param_changed, this);
            _panel->tau_green->callback(on_param_changed, this);
            _panel->tau_blue->callback(on_param_changed, this);
            _panel->old_color->callback(on_color_change, this);
            _panel->new_color->callback(on_color_change, this);

            panel = _panel;
        }
        break;
    }

    if(m_param_panel)
        delete m_param_panel;
    m_param_panel = panel;

    m_param_group->end();

    // must reposition here. if we set px,py in panel's constructor, it
    // won't be where we want (FLTK's quirkiness...) 
    if(panel)
        panel->position(px,py);

    // must redraw to make the panel appear on screen
    redraw();

    // update the image with the new effect
    update_image();
}

void MainFrame::on_apply_effect()
{
}

void MainFrame::on_undo_effect()
{
}

void MainFrame::update_image()
{
    // signal the rende thread that it has a new render job to do
    rod::unique_lock lk(m_mtx_render_data);

    m_has_new_render_job = true;

    m_wakeup.signal();
}

int main(int argc, char *argv[])
{
    try
    {
        // enables FLTK's multithreading facilities
        Fl::lock();

        Fl::visual(FL_RGB);

        MainFrame frame;
        frame.show();

        // got a command line parameter?
        if(argc == 2)
            frame.open(argv[1]); // treat it as a file to be opened

        // run main loop
        return Fl::run();
    }
    catch(std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}

/*{{{ Proxy event handlers */
#define CATCH() \
    catch(std::exception &e) \
{ \
    fl_alert("%s",e.what()); \
} \
catch(...) \
{ \
    fl_alert("Unknown error"); \
}

namespace {
    MainFrame *get_frame(Fl_Widget *w)
    {
        if(MainFrame *frame = dynamic_cast<MainFrame *>(w))
            return frame;
        else
        {
            assert(w);
            return get_frame(w->parent());
        }
    }
}

void on_file_open(Fl_Menu_ *m, void *)/*{{{*/
{
    try
    {
        get_frame(m)->on_file_open();
    }
    CATCH()
}/*}}}*/
void on_file_save(Fl_Menu_ *m, void *)/*{{{*/
{
    try
    {
        get_frame(m)->on_file_save();
    }
    CATCH()
}/*}}}*/
void on_file_save_as(Fl_Menu_ *m, void *)/*{{{*/
{
    try
    {
        get_frame(m)->on_file_save_as();
    }
    CATCH()
}/*}}}*/
void on_file_exit(Fl_Menu_ *m, void *)/*{{{*/
{
    try
    {
        get_frame(m)->on_file_exit();
    }
    CATCH()
}/*}}}*/

void on_choose_effect(Fl_Choice *m, void*)/*{{{*/
{
    try
    {
        get_frame(m)->on_choose_effect((effect_type)(ptrdiff_t)m->mvalue()->user_data_);
    }
    CATCH()
}/*}}}*/
void on_apply_effect(Fl_Return_Button *m, void*)/*{{{*/
{
    try
    {
        get_frame(m)->on_apply_effect();
    }
    CATCH()
}/*}}}*/
void on_undo_effect(Fl_Button *m, void*)/*{{{*/
{
    try
    {
        get_frame(m)->on_undo_effect();
    }
    CATCH()
}/*}}}*/
void on_change_grayscale(Fl_Light_Button*lb, void*)
{
    try
    {
        get_frame(lb)->on_change_grayscale(lb->value());
    }
    CATCH()
}
/*}}}*/
