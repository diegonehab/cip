#include <FL/fl_ask.H> // for fl_alert
#include <FL/Fl_File_Chooser.H>
#include <FL/Fl_Color_Chooser.H>
#include <vector>
#include <cuda_gl_interop.h>
#include <unistd.h> // for getopt
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

    void open(const std::string &fname);

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
    m_effects->add("Gradient edge detection",0,NULL,(void*)EFFECT_GRADIENT_EDGE_DETECTION);
    m_effects->add("Laplacian",0,NULL,(void*)EFFECT_LAPLACIAN);
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

// setup recursive filters and other stuff
void setup_recursive_filter(int width, int height, int rowstride)
{
    static bool init_noise = false;
    static int cur_width=-1, cur_height=-1, cur_rowstride=-1;
    static Vector<float,1+1> weights;

    if(!init_noise)
    {
        // calculate cubic b-spline weights
        float a = 2.f-std::sqrt(3.0f);

        weights[0] = 1+a;
        weights[1] = a;

        init_blue_noise();
        init_noise = true;
    }

    if(cur_width != width || cur_height!=height || cur_rowstride!=rowstride)
    {
        recursive_filter_5_setup(width,height,rowstride,
                                 weights, CLAMP_TO_EDGE, 1);

        cur_width = width;
        cur_height = height;
        cur_rowstride = rowstride;
    }
}

enum filter_flags
{
    VERBOSE=1
};

template <class T, class U, int C>
void call_filter(dimage_ptr<T,C> out, 
                 dimage_ptr<U,C> in,
                 const filter_operation &op,
                 int flags=0)
{
    int imgsize = in.width()*in.height();

    base_timer *timerzao = NULL, *timer = NULL;
    if(flags & VERBOSE)
        timerzao = &timers.cpu_add("Filter",imgsize,"P");

    // convolve with a bpsline3^-1 to make a cardinal post-filter
    if(flags & VERBOSE)
        timer = &timers.cpu_add("bspline3^-1 convolution",imgsize,"P");

    for(int i=0; i<C; ++i)
        recursive_filter_5(out[i], in[i]);

    if(timer)
        timer->stop();

    // do actual filtering
    if(flags & VERBOSE)
        timer = &timers.cpu_add("supersampling and transform",imgsize,"P");
    filter(out, op);

    if(timer)
        timer->stop();

    // convolve with a bpsline3^-1 to make a cardinal pre-filter
    if(flags & VERBOSE)
        timer = &timers.cpu_add("bspline3^-1 convolution",imgsize,"P");

    for(int i=0; i<C; ++i)
        recursive_filter_5(out[i]);

    if(timer)
        timer->stop();

    if(timerzao)
        timerzao->stop();

    if(flags & VERBOSE)
        timers.flush();
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
                call_filter(imgframe->get_grayscale_output(),
                            imgframe->get_grayscale_input(), op);
            }
            else
            {
                call_filter(imgframe->get_output(),
                            imgframe->get_input(), op);
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

            int imgsize = imgframe->get_output().width() *
                          imgframe->get_output().height();

            float rate = imgsize / timer.elapsed();
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


void MainFrame::open(const std::string &fname)
{
    std::vector<uchar4> imgdata;
    int width, height;

    load_image(fname, &imgdata, &width, &height);

    // create the image frame with the image data
    if(!m_image_frame)
    {
        m_image_frame = new ImageFrame(&imgdata[0], width, height);
        m_image_frame->show();
    }
    else
        m_image_frame->set_input_image(&imgdata[0], width, height);

    m_image_frame->copy_label(fname.c_str());

    setup_recursive_filter(width, height, m_image_frame->get_input().rowstride());
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
    default:
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
        break;
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

filter_operation parse_filter_operation(const std::string &spec)
{
    std::istringstream ss(spec);

    std::string opname;
    getline(ss,opname,'[');
    if(!ss || ss.eof() || opname.empty())
        throw std::runtime_error("Syntax error on effect specification");

    filter_operation op;

    if(opname == "identity")
        op.type = EFFECT_IDENTITY;
    else if(opname == "gradient_edge_detection")
        op.type = EFFECT_GRADIENT_EDGE_DETECTION;
    else if(opname == "laplacian")
        op.type = EFFECT_LAPLACIAN;
    else if(opname == "posterize")
    {
        op.type = EFFECT_POSTERIZE;
        ss >> op.levels;
    }
    else if(opname == "scale")
    {
        op.type = EFFECT_SCALE;
        ss >> op.scale;
    }
    else if(opname == "bias")
    {
        op.type = EFFECT_BIAS;
        ss >> op.bias;
    }
    else if(opname == "root")
    {
        op.type = EFFECT_ROOT;
        ss >> op.degree;
    }
    else if(opname == "threshold")
    {
        op.type = EFFECT_THRESHOLD;
        ss >> op.threshold;
    }
    else if(opname == "replacement")
    {
        op.type = EFFECT_REPLACEMENT;
        char c[8];
        ss >> op.old_color.x >> c[0] >> op.old_color.y >> c[1] >> op.old_color.z
           >> c[2] >> op.new_color.x >> c[3] >> op.new_color.y >> c[4] >> op.new_color.z
           >> c[5] >> op.tau.x >> c[6] >> op.tau.y >> c[7] >> op.tau.z;
        if(ss)
        {
            for(int i=0; i<8; ++i)
            {
                if(c[i] != ',')
                    ss.setstate(std::ios::failbit);
            }
        }
    }

    if(!ss || ss.get()!=']' || (ss.get(),!ss.eof()))
        throw std::runtime_error("Syntax error on effeect specification");

    return op;
}

void print_help(const char *progname)
{
    std::cout << "Usage: " << progname << "  [-e effect_descr] [-o output_file] [input_file]\n"
            " where effect_descr is one of:\n"
            "  - identity[]\n"
            "  - posterize[levels]\n"
            "  - scale[value]\n"
            "  - bias[value]\n"
            "  - root[degree]\n"
            "  - threshold[value]\n"
            "  - replacement[old_r,old_g,old_b,new_r,new_g,new_b,tau_r,tau_g,tau_b]\n"
            "  - gradient_edge_detection[]\n"
            "  - laplacian[]\n"
            "\n"
            "without -o, shows a GUI\n";
}

int main(int argc, char *argv[])
{
    try
    {
        std::string infile,
                    outfile,
                    effect = "identity[]";
        bool do_grayscale = false;

        int opt;
        while((opt = getopt(argc, argv, "-hgo:e:")) != -1)
        {
            switch(opt)
            {
            case 'h':
                print_help(basename(argv[0]));
                return 0;
            case 'o':
                outfile = optarg;
                break;
            case 'e':
                effect = optarg;
                break;
            case 'g':
                do_grayscale = true;
                break;
            case 1:
                if(!infile.empty())
                    throw std::runtime_error("Bad command line parameter (-h for help");
                infile = optarg;
                break;
            }
        }

        if(opt != -1)
            throw std::runtime_error("Bad command line parameter (-h for help");


        if(!outfile.empty())
        {
            if(infile.empty())
                throw std::runtime_error("Must specify an input image");

            filter_operation op = parse_filter_operation(effect);

            std::vector<uchar4> imgdata;
            int width, height;
            load_image(infile, &imgdata, &width, &height);

            dimage<uchar3,1> d_img;
            d_img.copy_from_host(imgdata, width, height);

            setup_recursive_filter(width, height, d_img.rowstride());

            if(do_grayscale)
            {
                dimage<float,1> d_gray;
                d_gray.resize(d_img.width(), d_img.height());
                grayscale(d_gray, &d_img);

                call_filter(&d_gray, &d_gray, op, VERBOSE);

                convert(&d_img, &d_gray);
            }
            else
            {
                dimage<float,3> d_channels;
                d_channels.resize(d_img.width(), d_img.height());

                convert(&d_channels, &d_img);

                call_filter(&d_channels, &d_channels, op, VERBOSE);

                convert(&d_img, &d_channels);
            }

            d_img.copy_to_host(imgdata);

            save_image(outfile, imgdata, width, height);
            return 0;
        }
        else
        {
            // enables FLTK's multithreading facilities
            Fl::lock();

            Fl::visual(FL_RGB);

            MainFrame frame;
            frame.show();

            // must open an image immediately?
            if(!infile.empty())
                frame.open(infile); // do it

            // run main loop
            return Fl::run();
        }
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
