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

#ifndef TIMER_H
#define TIMER_H

#include <string>
#include <list>
#include <cuda_runtime.h>

class base_timer/*{{{*/
{
public:
    base_timer(const char *type_label, size_t data_size=0, 
               const std::string &unit="");

    void start();
    void stop();
    float elapsed();
    bool is_stopped() const { return !m_started; }
    size_t data_size() const { return m_data_size; }
    const std::string &unit() const { return m_unit; }

    const char *type_label() { return m_type_label; }

protected:
    virtual void do_start() = 0;
    virtual void do_stop() = 0;
    virtual float do_get_elapsed() const = 0;

private:
    base_timer(const base_timer &);
    base_timer &operator=(const base_timer &);

    const char *m_type_label;
    float m_elapsed;
    bool m_started;
    size_t m_data_size;
    std::string m_unit;
};/*}}}*/

class gpu_timer : public base_timer/*{{{*/
{
public:
    gpu_timer(size_t data_size=0, const std::string &unit="", bool start=true);
    ~gpu_timer();

private:
    cudaEvent_t m_start, m_stop;

    virtual void do_start();
    virtual void do_stop();
    virtual float do_get_elapsed() const;
};/*}}}*/

class cpu_timer : public base_timer/*{{{*/
{
public:
    cpu_timer(size_t data_size=0, const std::string &unit="", bool start=true);
    ~cpu_timer() {}

private:
    double m_start_time, m_stop_time;

    virtual void do_start();
    virtual void do_stop();
    virtual float do_get_elapsed() const;

    double get_cpu_time() const;
};/*}}}*/

class scoped_timer_stop
{
public:
    scoped_timer_stop(base_timer &timer);
    ~scoped_timer_stop() { stop(); }

    void stop();

    float elapsed() const { return m_timer->elapsed(); }

private:
    base_timer *m_timer;
    static int m_global_padding;
};

class timer_pool
{
public:
    ~timer_pool() { }

    gpu_timer &gpu_add(const std::string &label, size_t data_size=0,
                       const std::string &unit="", bool start=true);
    cpu_timer &cpu_add(const std::string &label, size_t data_size=0,
                       const std::string &unit="", bool start=true);
    void flush();

private:
    struct timer_data
    {
        base_timer *timer;
        std::string label;
        int level;
    };

    typedef std::list<timer_data> timer_list;
    timer_list m_timers;
};

extern timer_pool timers;

#endif
