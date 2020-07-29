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

#include <FL/fl_ask.H> // for fl_alert
#include <FL/Fl_File_Chooser.H>
#include <FL/Fl_Color_Chooser.H>
#include <vector>
#include <cuda_gl_interop.h>
#include <getopt.h> // for getopt
#include "image_util.h"
#include "timer.h"
#include "filter.h"
#include "config.h"
#include "nlfilter_gui.h"
#include "image_frame.h"
#include "threads.h"
#if CUDA_SM < 20
#   include "cuPrintf.cuh"
#endif

class MainFrame : public MainFrameUI/*{{{*/
{
public:
    MainFrame();
    ~MainFrame();

    void open(const std::string &fname);

    void on_file_open();
    void on_file_save();
    void on_file_save_as();
    void on_file_exit();
    void on_window_point_sampling(bool enable);

    void on_choose_effect(effect_type effect);
    void on_change_grayscale(bool gs);
    void on_change_point_sampling(bool gs);
    void on_change_original(bool en);

    static void on_filter_changed(Fl_Widget *,MainFrame *frame);

    static void on_param_changed(Fl_Widget *,MainFrame *frame)
    {
        // update the image to reflect the parameter change
        frame->update_image();
    }

    void show_zoom_frame(int ww, int wh, int x,int y,int w,int h);

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

        if(m_image_frame_box)
            m_image_frame_box->show();
        if(m_zoom_frame)
            m_zoom_frame->show();
        if(m_zoom_frame_box)
            m_zoom_frame_box->show();
    }
    virtual void hide()
    {
        if(m_image_frame)
            m_image_frame->hide();
        if(m_image_frame_box)
            m_image_frame_box->hide();
        if(m_zoom_frame)
            m_zoom_frame->hide();
        if(m_zoom_frame_box)
            m_zoom_frame_box->hide();
        MainFrameUI::hide();
    }

private:
    std::string m_file_name;
    bool m_use_point_sampling;

    void restart_render_thread()
    {
        stop_render_thread();
        start_render_thread();
    }

    void start_render_thread();
    void stop_render_thread();

    void update_image();

    filter_operation get_filter_operation() const;

    static void *render_thread(MainFrame *frame);

    rod::thread m_render_thread;
    rod::condition_variable m_wakeup; // wakes up the render thread
    rod::mutex m_mtx_render_data;
    bool m_terminate_thread; // tells the render thread to finish its business
    bool m_has_new_render_job;
    bool m_show_original_image;

    Fl_Group *m_param_panel; // current effect parameter panel
    ImageFrame *m_image_frame; 

    rod::mutex m_mtx_imgframe_box;
    ImageFrame *m_image_frame_box; 

    ImageFrame *m_zoom_frame, *m_zoom_frame_box;
    int m_zoom_x, m_zoom_y, m_zoom_w, m_zoom_h;

    int m_last_imgframe_box_pos_x,
        m_last_imgframe_box_pos_y;
};/*}}}*/

MainFrame::MainFrame()/*{{{*/
    : MainFrameUI(753,319,720,155, "NLFilter")
    , m_render_thread((void(*)(void*))&MainFrame::render_thread, this, false)
    , m_param_panel(NULL)
    , m_image_frame(NULL)
    , m_image_frame_box(NULL)
    , m_terminate_thread(false)
    , m_has_new_render_job(false)
    , m_last_imgframe_box_pos_x(-666) // our "not set" value \m/
    , m_last_imgframe_box_pos_y(-666)
    , m_use_point_sampling(false)
    , m_show_original_image(false)
    , m_zoom_frame(NULL)
    , m_zoom_frame_box(NULL)
{
    m_zoom_x = m_zoom_y = m_zoom_w = m_zoom_h = 0;

    m_effects->add("Identity",0,NULL,(void*)EFFECT_IDENTITY);
    m_effects->add("Posterize",0,NULL,(void*)EFFECT_POSTERIZE);
    m_effects->add("Scale",0,NULL,(void*)EFFECT_SCALE);
    m_effects->add("Bias",0,NULL,(void*)EFFECT_BIAS);
    m_effects->add("Root",0,NULL,(void*)EFFECT_ROOT);
    m_effects->add("Threshold",0,NULL,(void*)EFFECT_THRESHOLD);
    m_effects->add("Replacement",0,NULL,(void*)EFFECT_REPLACEMENT);
    m_effects->add("Gradient edge detection",0,NULL,(void*)EFFECT_GRADIENT_EDGE_DETECTION);
    m_effects->add("Laplacian",0,NULL,(void*)EFFECT_LAPLACIAN);
    m_effects->add("Laplace edge enhancement",0,NULL,(void*)EFFECT_LAPLACE_EDGE_ENHANCEMENT);
    m_effects->add("Yaroslavski bilateral",0,NULL,(void*)EFFECT_YAROSLAVSKY_BILATERAL);
    m_effects->add("Brightness and contrast",0,NULL,(void*)EFFECT_BRIGHTNESS_CONTRAST);
    m_effects->add("Hue, saturation and lightness",0,NULL,(void*)EFFECT_HUE_SATURATION_LIGHTNESS);
    m_effects->add("Unsharp mask",0,NULL,(void*)EFFECT_UNSHARP_MASK);
    m_effects->add("Bilateral",0,NULL,(void*)EFFECT_BILATERAL);
    m_effects->add("Emboss",0,NULL,(void*)EFFECT_EMBOSS);
    m_effects->value(0);

    m_pre_filter->add("Cubic BSpline",0,NULL,(void*)FILTER_BSPLINE3);
    m_pre_filter->add("Cardinal Cubic BSpline",0,NULL,(void*)FILTER_CARDINAL_BSPLINE3);
    m_pre_filter->add("Michell-Netravali",0,NULL,(void*)FILTER_MITCHELL_NETRAVALI);
    m_pre_filter->add("Box",0,NULL,(void*)FILTER_BOX);
    m_pre_filter->value(1);
    m_pre_filter->callback((Fl_Callback *)&MainFrame::on_filter_changed, this);

    m_post_filter->add("Cubic BSpline",0,NULL,(void*)FILTER_BSPLINE3);
    m_post_filter->add("Cardinal Cubic BSpline",0,NULL,(void*)FILTER_CARDINAL_BSPLINE3);
    m_post_filter->add("Michell-Netravali",0,NULL,(void*)FILTER_MITCHELL_NETRAVALI);
    m_post_filter->add("Box",0,NULL,(void*)FILTER_BOX);
    m_post_filter->add("Sacht-Nehab3",0,NULL,(void*)FILTER_SACHT_NEHAB3);
    m_post_filter->value(1);
    m_post_filter->callback((Fl_Callback *)&MainFrame::on_filter_changed, this);
}/*}}}*/
MainFrame::~MainFrame()/*{{{*/
{
    stop_render_thread();
}/*}}}*/

filter_operation MainFrame::get_filter_operation() const/*{{{*/
{
    filter_operation op;

    op.use_supersampling = true;

    op.type = (effect_type)(ptrdiff_t)m_effects->mvalue()->user_data_;

    op.pre_filter = (filter_type)(ptrdiff_t)m_pre_filter->mvalue()->user_data_;
    op.post_filter = (filter_type)(ptrdiff_t)m_post_filter->mvalue()->user_data_;


    if(const ParamPosterizeUI *panel = dynamic_cast<const ParamPosterizeUI *>(m_param_panel))
        op.levels = std::max<int>(2,panel->levels->value());
    else if(const ParamScaleUI *panel = dynamic_cast<const ParamScaleUI *>(m_param_panel))
        op.scale = panel->scale->value();
    else if(const ParamBiasUI *panel = dynamic_cast<const ParamBiasUI *>(m_param_panel))
        op.bias = panel->bias->value();
    else if(const ParamRootUI *panel = dynamic_cast<const ParamRootUI *>(m_param_panel))
        op.degree = panel->degree->value();
    else if(const ParamThresholdUI *panel = dynamic_cast<const ParamThresholdUI *>(m_param_panel))
    {
        op.minimum = panel->minimum->value();
        op.maximum = panel->maximum->value();
    }
    else if(const ParamLaplaceEdgeEnhancementUI *panel = dynamic_cast<const ParamLaplaceEdgeEnhancementUI *>(m_param_panel))
        op.multiple = panel->multiple->value();
    else if(const ParamYaroslavskyBilateralUI *panel = dynamic_cast<const ParamYaroslavskyBilateralUI *>(m_param_panel))
    {
        op.rho = panel->rho->value();
        op.h = panel->h->value();
    }
    else if(const ParamBrightnessContrastUI *panel = dynamic_cast<const ParamBrightnessContrastUI *>(m_param_panel))
    {
        op.brightness = panel->brightness->value();
        op.contrast = panel->contrast->value();
    }
    else if(const ParamHueSaturationLightnessUI *panel = dynamic_cast<const ParamHueSaturationLightnessUI *>(m_param_panel))
    {
        op.hue = panel->hue->value();
        op.saturation = panel->saturation->value();
        op.lightness = panel->lightness->value();
    }
    else if(const ParamReplacementUI *panel = dynamic_cast<const ParamReplacementUI *>(m_param_panel))
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
    else if(const ParamUnsharpMaskUI *panel = dynamic_cast<const ParamUnsharpMaskUI *>(m_param_panel))
    {
        op.sigma = panel->radius->value();
        op.amount = panel->amount->value();
        op.threshold = panel->threshold->value();
    }
    else if(const ParamBilateralUI *panel = dynamic_cast<const ParamBilateralUI *>(m_param_panel))
    {
        op.sigma_r = panel->sigma_r->value();
        op.sigma_s = panel->sigma_s->value();
    }
    else if(const ParamEmbossUI *panel = dynamic_cast<const ParamEmbossUI *>(m_param_panel))
    {
        op.amount = panel->amount->value();
        op.offset = panel->offset->value();
    }

    return op;
}/*}}}*/

void MainFrame::start_render_thread()/*{{{*/
{
    if(!m_render_thread.is_started())
    {
        m_terminate_thread = false;
        m_render_thread.start();
    }
}/*}}}*/
void MainFrame::stop_render_thread()/*{{{*/
{
    // signal the render thread to terminate
    rod::unique_lock lk(m_mtx_render_data);
    m_terminate_thread = true;
    m_wakeup.signal();
    lk.unlock();

    // just wait until it finishes
    m_render_thread.join();
}/*}}}*/

// defined on timer.cpp
std::string unit_value(double v, double base);

void show_error(std::string *data)/*{{{*/
{
    if(data != NULL)
    {
        fl_alert("%s",data->c_str());
        delete data;
    }
}/*}}}*/

void *MainFrame::render_thread(MainFrame *frame)/*{{{*/
{
    filter_plan *plan = NULL, *plan_box = NULL;

    try
    {
        rod::unique_lock lkiframebox(frame->m_mtx_imgframe_box);

        ImageFrame *imgframe = frame->m_image_frame,
                   *imgframe_box = frame->m_image_frame_box;

        filter_operation op = frame->get_filter_operation(),
                         op_box = op;
        op_box.use_supersampling = false;
        op_box.pre_filter = op_box.post_filter = FILTER_BOX;

        if(frame->m_use_point_sampling)
        {
            op.use_supersampling = false;
            op.pre_filter = op_box.post_filter = FILTER_BOX;
        }

        bool grayscale = frame->m_grayscale->value();

        if(grayscale)
        {
            plan = filter_create_plan(imgframe->get_grayscale_input(), op);
            if(imgframe_box)
                plan_box = filter_create_plan(imgframe_box->get_grayscale_input(), op_box);
        }
        else
        {
            plan = filter_create_plan(imgframe->get_input(), op);
            if(imgframe_box)
                plan_box = filter_create_plan(imgframe_box->get_input(), op_box);
        }

        lkiframebox.unlock();

        while(!frame->m_terminate_thread)
        {
            rod::unique_lock lk(frame->m_mtx_render_data);

            // no render job? sleep
            if(imgframe == NULL || !frame->m_has_new_render_job)
                frame->m_wakeup.wait(lk);

            // maybe it was set during wait
            imgframe = frame->m_image_frame;

            if(!frame->m_terminate_thread &&
               imgframe!=NULL && frame->m_has_new_render_job)
            {
                // fill the operation struct along with its parameters based
                // on what is set by the user
                frame->m_has_new_render_job = false;

                lk.unlock();

                double elapsed;

                // supersampling
                {
                    // lock buffers since we'll write on them
                    ImageFrame::OutputBufferLocker lkbuffers(*imgframe);

                    filter_operation op = frame->get_filter_operation();

                    if(frame->m_use_point_sampling)
                    {
                        op.use_supersampling = false;
                        op.pre_filter = op_box.post_filter = FILTER_BOX;
                    }

                    gpu_timer timer;

                    // just process one (grayscale) channel?
                    if(grayscale)
                        filter(plan, imgframe->get_grayscale_output(), op);
                    else
                        filter(plan, imgframe->get_output(), op);

                    timer.stop();
                    elapsed = timer.elapsed();

                    if(frame->m_zoom_frame)
                    {
                        if(grayscale)
                        {
                            subimage(frame->m_zoom_frame->get_grayscale_output(), 
                                     imgframe->get_grayscale_output(),
                                     frame->m_zoom_x,frame->m_zoom_y,
                                     frame->m_zoom_w, frame->m_zoom_h);
                        }
                        else
                        {
                            subimage(frame->m_zoom_frame->get_output(), 
                                     imgframe->get_output(),
                                     frame->m_zoom_x, frame->m_zoom_y,
                                     frame->m_zoom_w, frame->m_zoom_h);
                        }
                    }
                }

                lkiframebox.lock();

                imgframe_box = frame->m_image_frame_box;

                // box
                if(imgframe_box)
                {
                    // lock buffers since we'll write on them
                    ImageFrame::OutputBufferLocker lkbuffers(*imgframe_box);

                    filter_operation op = frame->get_filter_operation();
                    op.use_supersampling = false;
                    op.pre_filter = op.post_filter = FILTER_BOX;

//                    gpu_timer timer;

                    // just process one (grayscale) channel?
                    if(grayscale)
                        filter(plan_box, imgframe_box->get_grayscale_output(), op);
                    else
                        filter(plan_box, imgframe_box->get_output(), op);

                    if(frame->m_zoom_frame_box)
                    {
                        if(grayscale)
                        {
                            subimage(frame->m_zoom_frame_box->get_grayscale_output(), 
                                     imgframe_box->get_grayscale_output(),
                                     frame->m_zoom_x,frame->m_zoom_y,
                                     frame->m_zoom_w, frame->m_zoom_h);
                        }
                        else
                        {
                            subimage(frame->m_zoom_frame_box->get_output(), 
                                     imgframe_box->get_output(),
                                     frame->m_zoom_x, frame->m_zoom_y,
                                     frame->m_zoom_w, frame->m_zoom_h);
                        }
                    }

  //                  timer.stop();
                }
                lkiframebox.unlock();

                // avoids a deadlock in Fl::lock() because usually we're
                // destroying the main window if terminate_thread is true
                // at this point
                if(frame->m_terminate_thread)
                    break;

                // tell the image window to update its content and update
                // performance counters
                Fl::lock();

                frame->m_status_fps->value(1.0/elapsed);

                int imgsize = imgframe->get_output().width() *
                              imgframe->get_output().height();

                float rate = imgsize / elapsed;
                std::string srate = unit_value(rate, 1000), unit;
                {
                    std::istringstream ss(srate);
                    ss >> rate >> unit;
                    unit += "P/s ";
                }
                frame->m_status_rate->copy_label(unit.c_str());
                frame->m_status_rate->value(rate);

                imgframe->swap_buffers();

                lkiframebox.lock();

                imgframe_box = frame->m_image_frame_box;

                if(imgframe_box)
                    imgframe_box->swap_buffers();

                if(frame->m_zoom_frame_box)
                    frame->m_zoom_frame_box->swap_buffers();

                if(frame->m_zoom_frame)
                    frame->m_zoom_frame->swap_buffers();

                lkiframebox.unlock();

                Fl::unlock();
                Fl::awake((void *)NULL); // awake message loop processing
            }
        }
    }
    catch(std::exception &e)
    {
        std::ostringstream ss;
        ss << "Render thread error: " << e.what();

        Fl::awake((Fl_Awake_Handler)&show_error, new std::string(ss.str()));
    }
    catch(...)
    {
        Fl::awake((Fl_Awake_Handler)&show_error, new std::string("Render thread error: unknown"));
    }

    free(plan);
    free(plan_box);
}/*}}}*/

void MainFrame::on_change_grayscale(bool gs)/*{{{*/
{

    if(m_image_frame)
        m_image_frame->set_grayscale(gs);

    if(m_image_frame_box)
        m_image_frame_box->set_grayscale(gs);

    if(m_zoom_frame)
        m_zoom_frame->set_grayscale(gs);

    if(m_zoom_frame_box)
        m_zoom_frame_box->set_grayscale(gs);

    // must preprocess input image again
    if(!m_show_original_image)
        restart_render_thread();
    update_image();
}/*}}}*/
void MainFrame::on_change_point_sampling(bool ps)/*{{{*/
{ 
    m_use_point_sampling = ps;

    if(m_use_point_sampling)
        m_image_frame->copy_label(("Point Sampling: "+m_file_name).c_str());
    else
        m_image_frame->copy_label(("Supersampling: "+m_file_name).c_str());

    if(!m_show_original_image)
        restart_render_thread();
    update_image();
}/*}}}*/
void MainFrame::on_change_original(bool gs)/*{{{*/
{
    m_show_original_image = gs;

    if(m_show_original_image)
    {
        stop_render_thread();
        m_image_frame->reset();
        if(m_image_frame_box)
            m_image_frame_box->reset();

        m_status_rate->value(0);
        m_status_fps->value(0);
    }
    else
    {
        start_render_thread();
        update_image();
    }
}/*}}}*/

void MainFrame::on_filter_changed(Fl_Widget *,MainFrame *frame)/*{{{*/
{
    // must reprocess input image with new pre- and post-filter
    frame->restart_render_thread();
    frame->update_image();
}/*}}}*/

void MainFrame::open(const std::string &fname)/*{{{*/
{
    std::vector<uchar4> imgdata;
    int width, height;

    load_image(fname, &imgdata, &width, &height);

    // create the image frame with the image data
    if(!m_image_frame)
        m_image_frame = new ImageFrame(0,0,width,height);

    dimage<uchar3> img_byte3;
    img_byte3.copy_from_host(&imgdata[0], width, height);

    dimage<float3> img_float3(width, height);
    convert(&img_float3, &img_byte3);

    m_image_frame->set_input_image(&img_float3);
    m_image_frame->copy_label(("Supersampling: "+fname).c_str());

    restart_render_thread();

    m_file_name = fname;
}/*}}}*/

void MainFrame::show_zoom_frame(int ww, int wh, int x,int y,int w,int h)/*{{{*/
{
    m_zoom_x = x;
    m_zoom_y = y;
    m_zoom_w = w;
    m_zoom_h = h;

    m_zoom_frame = new ImageFrame(0,0,ww,wh);
    m_zoom_frame->copy_label("Supersample Zoom");
    m_zoom_frame->show();
    m_zoom_frame->set_blank_input_image(w,h);

    m_zoom_frame_box = new ImageFrame(ww+5,0,ww,wh);
    m_zoom_frame_box->copy_label("Point Sampling Zoom");
    m_zoom_frame_box->show();
    m_zoom_frame_box->set_blank_input_image(w,h);
}/*}}}*/

void MainFrame::on_file_open()/*{{{*/
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
}/*}}}*/

void MainFrame::on_file_save()/*{{{*/
{
}/*}}}*/
void MainFrame::on_file_save_as()/*{{{*/
{
}/*}}}*/
void MainFrame::on_file_exit()/*{{{*/
{
    hide();
}/*}}}*/

void MainFrame::on_window_point_sampling(bool enable)/*{{{*/
{
    // fltk is nuts, we can't trust on m->check() being
    // correct during user selection. so let's just toggle
    // current state
    enable = m_image_frame_box==NULL;

    rod::unique_lock lk(m_mtx_imgframe_box);

    if(!enable)
    {
        m_last_imgframe_box_pos_x = m_image_frame_box->x();
        m_last_imgframe_box_pos_y = m_image_frame_box->y();

        assert(m_image_frame_box != NULL);
        // we can't hide the window, deinit_gl will fail if we do
        //m_image_frame_box->hide();
        delete m_image_frame_box;
        m_image_frame_box = NULL;
    }
    else if(m_image_frame)
    {
        assert(m_image_frame_box == NULL);

        if(m_last_imgframe_box_pos_x != -666)
        {
            m_image_frame_box = new ImageFrame(m_last_imgframe_box_pos_x,
                                               m_last_imgframe_box_pos_y,
                                               m_image_frame->w(),
                                               m_image_frame->h());
        }
        else
        {
            m_image_frame_box = new ImageFrame(
                                m_image_frame->x()+m_image_frame->w()+10,
                                m_image_frame->y(),
                                m_image_frame->w(),
                                m_image_frame->h());
        }

        m_image_frame_box->set_input_image(dimage_ptr<const float,3>(m_image_frame->get_input()));
        m_image_frame_box->copy_label(("Point Sampling: "+m_file_name).c_str());
    }

    lk.unlock();

    restart_render_thread();
}/*}}}*/

void MainFrame::on_choose_effect(effect_type effect)/*{{{*/
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
            _panel->minimum->callback(on_param_changed, this);
            _panel->maximum->callback(on_param_changed, this);
            panel = _panel;
        }
        break;
    case EFFECT_LAPLACE_EDGE_ENHANCEMENT:
        {
            ParamLaplaceEdgeEnhancementUI *_panel 
                = new ParamLaplaceEdgeEnhancementUI(0,0,pw,ph);
            _panel->multiple->callback(on_param_changed, this);
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
    case EFFECT_YAROSLAVSKY_BILATERAL:
        {
            ParamYaroslavskyBilateralUI *_panel 
                = new ParamYaroslavskyBilateralUI(0,0,pw,ph);
            _panel->rho->callback(on_param_changed, this);
            _panel->h->callback(on_param_changed, this);
            panel = _panel;
        }
        break;
    case EFFECT_BRIGHTNESS_CONTRAST:
        {
            ParamBrightnessContrastUI *_panel 
                = new ParamBrightnessContrastUI(0,0,pw,ph);
            _panel->brightness->callback(on_param_changed, this);
            _panel->contrast->callback(on_param_changed, this);
            panel = _panel;
        }
        break;
    case EFFECT_HUE_SATURATION_LIGHTNESS:
        {
            ParamHueSaturationLightnessUI *_panel 
                = new ParamHueSaturationLightnessUI(0,0,pw,ph);
            _panel->hue->callback(on_param_changed, this);
            _panel->saturation->callback(on_param_changed, this);
            _panel->lightness->callback(on_param_changed, this);
            panel = _panel;
        }
        break;
    case EFFECT_UNSHARP_MASK:
        {
            ParamUnsharpMaskUI *_panel 
                = new ParamUnsharpMaskUI(0,0,pw,ph);
            _panel->radius->callback(on_param_changed, this);
            _panel->amount->callback(on_param_changed, this);
            _panel->threshold->callback(on_param_changed, this);
            panel = _panel;

            // need to create auxiliar texture with blurred luminance
            restart_render_thread();
        }
        break;
    case EFFECT_BILATERAL:
        {
            ParamBilateralUI *_panel 
                = new ParamBilateralUI(0,0,pw,ph);
            _panel->sigma_r->callback(on_param_changed, this);
            _panel->sigma_s->callback(on_param_changed, this);
            panel = _panel;
        }
        break;
    case EFFECT_EMBOSS:
        {
            ParamEmbossUI *_panel 
                = new ParamEmbossUI(0,0,pw,ph);
            _panel->amount->callback(on_param_changed, this);
            _panel->offset->callback(on_param_changed, this);
            panel = _panel;
        }
        break;
    }

    if(m_param_panel)
    {
        m_param_panel->hide();
        m_param_group->remove(*m_param_panel);
        delete m_param_panel;
    }
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
}/*}}}*/

void MainFrame::update_image()/*{{{*/
{
    // signal the rende thread that it has a new render job to do
    rod::unique_lock lk(m_mtx_render_data);

    m_has_new_render_job = true;

    m_wakeup.signal();
}/*}}}*/

filter_operation parse_filter_operation(const std::string &spec)/*{{{*/
{
    std::istringstream ss(spec);

    std::string opname;
    getline(ss,opname,'[');
    if(!ss || ss.eof() || opname.empty())
        throw std::runtime_error("Syntax error on effect specification");

    filter_operation op;

    op.use_supersampling = true;

    if(opname == "identity")
        op.type = EFFECT_IDENTITY;
    else if(opname == "gradient_edge_detection")
        op.type = EFFECT_GRADIENT_EDGE_DETECTION;
    else if(opname == "laplace_edge_enhancement")
    {
        op.type = EFFECT_LAPLACE_EDGE_ENHANCEMENT;
        ss >> op.multiple;
    }
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
        char c;
        ss >> op.minimum >> c >> op.maximum;
        if(c != ',')
            ss.setstate(std::ios::failbit);
    }
    else if(opname == "yaroslavsky_bilateral")
    {
        op.type = EFFECT_YAROSLAVSKY_BILATERAL;
        char c;
        ss >> op.rho >> c >> op.h;
        if(c != ',')
            ss.setstate(std::ios::failbit);
    }
    else if(opname == "brightness_contrast")
    {
        op.type = EFFECT_BRIGHTNESS_CONTRAST;
        char c;
        ss >> op.brightness >> c >> op.contrast;
        if(c != ',')
            ss.setstate(std::ios::failbit);
    }
    else if(opname == "hue_saturation_lightness")
    {
        op.type = EFFECT_HUE_SATURATION_LIGHTNESS;
        char c[2];
        ss >> op.hue >> c[0] >> op.saturation >> c[1] >> op.lightness;
        if(c[0] != ',' || c[1] != ',')
            ss.setstate(std::ios::failbit);
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
    else if(opname == "unsharp_mask")
    {
        op.type = EFFECT_UNSHARP_MASK;
        char c[2];
        ss >> op.sigma >> c[0] >> op.amount >> c[1] >> op.threshold;
        if(c[0] != ',' || c[1] != ',')
            ss.setstate(std::ios::failbit);
    }
    else if(opname == "bilateral")
    {
        op.type = EFFECT_BILATERAL;
        char c;
        ss >> op.sigma_s >> c >> op.sigma_r;
        if(c != ',')
            ss.setstate(std::ios::failbit);
    }
    else if(opname == "emboss")
    {
        op.type = EFFECT_EMBOSS;
        char c;
        ss >> op.amount >> c >> op.offset;
        if(c != ',')
            ss.setstate(std::ios::failbit);
    }
    else 
        throw std::runtime_error("Bad effect type");

    if(!ss || ss.get()!=']' || (ss.get(),!ss.eof()))
        throw std::runtime_error("Syntax error on effect specification");

    return op;
}/*}}}*/

filter_type parse_filter_type(const std::string &name)/*{{{*/
{
    if(name == "bspline3")
        return FILTER_BSPLINE3;
    else if(name == "card-bspline3")
        return FILTER_CARDINAL_BSPLINE3;
    else if(name == "mitchell-netravali")
        return FILTER_MITCHELL_NETRAVALI;
    else if(name == "box")
        return FILTER_BOX;
    else if(name == "sacht-nehab3")
        return FILTER_SACHT_NEHAB3;
    else
        throw std::runtime_error("Bad filter type");
}/*}}}*/

void print_help(const char *progname)/*{{{*/
{
    std::cout << "Usage: " << progname << " [-v/--verbose] [--post post_filter] [--pre pre_filter] [-e/--effect effect_descr] [-o/--output output_file] [-h/--help] [input_file]\n"
            " where effect_descr is one of:\n"
            "  - identity[]\n"
            "  - posterize[levels]\n"
            "  - scale[value]\n"
            "  - bias[value]\n"
            "  - root[degree]\n"
            "  - threshold[min,max]\n"
            "  - replacement[old_r,old_g,old_b,new_r,new_g,new_b,tau_r,tau_g,tau_b]\n"
            "  - gradient_edge_detection[]\n"
            "  - laplacian[]\n"
            "  - laplace_edge_enhancement[mult]\n"
            "  - yaroslavsky_bilateral[rho,h]\n"
            "  - brightness_contrast[brightness,contrast]\n"
            "  - hue_saturation_lightness[hue,saturation,lightness]\n"
            "  - unsharp_mask[sigma,amount,threshold]\n"
            "  - bilateral[sigma_s,sigma_r]\n"
            "  - emboss[amount,offset]\n"
            "\n"
            " pre_filter and post_filter are:\n"
            "  - bspline3\n"
            "  - card-bspline3\n"
            "  - mitchell-netravali\n"
            "  - box\n"
            "\n"
            "without -o, shows a GUI\n";
}/*}}}*/

int main(int argc, char *argv[])/*{{{*/
{
    try
    {
        std::string infile,
                    outfile,
                    effect = "identity[]";
        std::string zoom_geom;
        bool do_grayscale = false;
        int flags = 0;
        filter_type prefilter = FILTER_CARDINAL_BSPLINE3,
                    postfilter = FILTER_CARDINAL_BSPLINE3;


        static struct option long_options[] = {
            {"pre",required_argument,0,0},
            {"post",required_argument,0,0},
            {"help",required_argument,0,'h'},
            {"effect",required_argument,0,'e'},
            {"output",required_argument,0,'o'},
            {"verbose",no_argument,0,'v'},
            {"zoom",required_argument,0,'z'},
            {0,0,0,0}
        };

        int opt, optindex;
        while((opt = getopt_long(argc, argv, "-hz:vgo:e:",
                                 long_options, &optindex)) != -1)
        {
            switch(opt)
            {
            case 0:
                if(long_options[optindex].name == std::string("pre"))
                    prefilter = parse_filter_type(optarg);
                else if(long_options[optindex].name == std::string("post"))
                    postfilter = parse_filter_type(optarg);
                break;
            case 'h':
                print_help(basename(argv[0]));
                return 0;
            case 'o':
                outfile = optarg;
                break;
            case 'e':
                effect = optarg;
                break;
            case 'z':
                zoom_geom = optarg;
                break;
            case 'v':
                flags |= VERBOSE;
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

#if CUDA_SM < 20
            cudaPrintfInit();
#endif


            filter_operation op = parse_filter_operation(effect);

            op.pre_filter = prefilter;
            op.post_filter = postfilter;

            std::vector<uchar4> imgdata;
            int width, height;
            load_image(infile, &imgdata, &width, &height);

            dimage<uchar3,1> d_img;
            d_img.copy_from_host(imgdata, width, height);

            base_timer *timerzao = &timers.gpu_add("Total Processing",
                                                  width*height, "P", false),
                       *timer = NULL;

            if(do_grayscale)
            {
                dimage<float,1> d_gray;
                d_gray.resize(d_img.width(), d_img.height());
                grayscale(&d_gray, &d_img);

                timerzao->start();

                timer = &timers.gpu_add("Preprocessing", width*height, "P");
                filter_plan *plan
                    = filter_create_plan(dimage_ptr<const float,1>(&d_gray), op,
                            flags);
                timer->stop();

                timer = &timers.gpu_add("Operation", width*height, "P");
                filter(plan, &d_gray, op);
                timer->stop();

                timerzao->stop();

                free(plan);

                convert(&d_img, &d_gray);
            }
            else
            {
                dimage<float,3> d_channels;
                d_channels.resize(d_img.width(), d_img.height());

                convert(&d_channels, &d_img);

                timerzao->start();

                timer = &timers.gpu_add("Preprocessing", width*height, "P");
                filter_plan *plan
                    = filter_create_plan(dimage_ptr<const float,3>(&d_channels), op,
                            flags);
                timer->stop();

                timer = &timers.gpu_add("Operation", width*height, "P");
                filter(plan, &d_channels, op);
                timer->stop();

                timerzao->stop();

                free(plan);

                convert(&d_img, &d_channels);
            }

            d_img.copy_to_host(imgdata);

            //save_image(outfile, imgdata, width, height);

            timers.flush();

#if CUDA_SM < 20
            cudaPrintfDisplay(stdout, true);
            cudaPrintfEnd();
#endif

            return 0;
        }
        else
        {
            // enables FLTK's multithreading facilities
            Fl::lock();

            Fl::visual(FL_RGB);

            MainFrame frame;
            frame.show();

            if(!zoom_geom.empty())
            {
                int x, y, w, h, ww, wh;
                int nread = sscanf(zoom_geom.c_str(), "%ux%u+%u+%u-%ux%u",&w,&h,&x,&y,&ww,&wh);
                if(nread != 6)
                    throw std::runtime_error("Bad zoom specification");
                frame.show_zoom_frame(ww,wh,x,y,w,h);
            }

            // must open an image immediately?
            if(!infile.empty())
                frame.open(infile); // do it

#if CUDA_SM < 20
            cudaPrintfInit();
#endif

            // run main loop
            return Fl::run();
        }
    }
    catch(std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}/*}}}*/

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
void on_window_point_sampling(Fl_Menu_ *m, void *)/*{{{*/
{
    try
    {
        get_frame(m)->on_window_point_sampling(m->value());
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
void on_change_grayscale(Fl_Light_Button*lb, void*)/*{{{*/
{
    try
    {
        get_frame(lb)->on_change_grayscale(lb->value());
    }
    CATCH()
}/*}}}*/
void on_change_point_sampling(Fl_Light_Button*lb, void*)/*{{{*/
{
    try
    {
        get_frame(lb)->on_change_point_sampling(lb->value());
    }
    CATCH()
}/*}}}*/
void on_change_original(Fl_Light_Button*lb, void*)/*{{{*/
{
    try
    {
        get_frame(lb)->on_change_original(lb->value());
    }
    CATCH()
}/*}}}*/
/*}}}*/
