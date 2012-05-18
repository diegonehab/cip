#include <pthread.h>
#include <sstream>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <cerrno>
#include "threads.h"

namespace
{
    void check_error(int err, const char *prefix)
    {
        if(err == 0)
            return;
        else
        {
            std::ostringstream ss;
            ss << prefix << ": " << std::strerror(err);
            throw std::runtime_error(ss.str());
        }
    }
}

namespace rod
{

// mutex ---------------------------------------------------------------

struct mutex::impl
{
    pthread_mutex_t mtx;
};

mutex::mutex()
{
    pimpl = new impl;
    try
    {
        check_error(pthread_mutex_init(&pimpl->mtx, NULL),"pthread_mutex_init");
    }
    catch(...)
    {
        delete pimpl;
        throw;
    }
}

mutex::~mutex()
{
    int err = pthread_mutex_destroy(&pimpl->mtx);
    assert(err == 0);
    delete pimpl;
}

void mutex::lock()
{
    check_error(pthread_mutex_lock(&pimpl->mtx), "pthread_mutex_lock");
}

void mutex::unlock()
{
    check_error(pthread_mutex_unlock(&pimpl->mtx), "pthread_mutex_unlock");
}

// Condition Variable ----------------------------------------------------

struct condition_variable::impl
{
    pthread_cond_t cond;
};

condition_variable::condition_variable()
{
    pimpl = new impl;
    try
    {
        check_error(pthread_cond_init(&pimpl->cond, NULL), "pthread_cond_init");
    }
    catch(...)
    {
        delete pimpl;
        throw;
    }
}

condition_variable::~condition_variable()
{
    int err = pthread_cond_destroy(&pimpl->cond);
    assert(err == 0);
    delete pimpl;
}

void condition_variable::wait(unique_lock &lk)
{
    check_error(pthread_cond_wait(&pimpl->cond, &lk.m_mtx.pimpl->mtx),
                "pthread_cond_wait");
}

void condition_variable::signal()
{
    check_error(pthread_cond_signal(&pimpl->cond), "pthread_cond_signal");
}

// thread -----------------------------------------------------------------

struct thread::impl
{
    pthread_t thread;
    pthread_attr_t attr;

    function_type user_thread_start;
    void *user_data;
    std::string error;
    bool is_running;

    static void *thread_start(impl *pimpl)
    {
        pimpl->is_running = true;
        try
        {
            pimpl->error = "";
            pimpl->user_thread_start(pimpl->user_data);
        }
        catch(std::exception &e)
        {
            pimpl->error = e.what();
        }
        catch(...)
        {
            pimpl->error = "Unexpected error";
        }

        pimpl->is_running = false;

        pthread_exit(NULL);
        return NULL;
    }
};

thread::thread(function_type thread_start, void *data, bool do_start)
    : m_started(false)
{
    pimpl = new impl;
    pimpl->user_thread_start = thread_start;
    pimpl->user_data = data;
    pimpl->is_running = false;

    bool attr_init = false;

    try
    {
        check_error(pthread_attr_init(&pimpl->attr),"pthread_attr_init");
        attr_init = true;
        check_error(pthread_attr_setdetachstate(&pimpl->attr, 
                                                PTHREAD_CREATE_JOINABLE),
                    "pthread_attr_set_detachstate");

        if(do_start)
            start();
    }
    catch(...)
    {
        if(attr_init)
            pthread_attr_destroy(&pimpl->attr);
        delete pimpl;
        throw;
    }
}

thread::~thread()
{
    join();
}

void thread::start()
{
    assert(!is_started());
    if(is_started())
        throw std::logic_error("Thread already started");

    check_error(pthread_create(&pimpl->thread, &pimpl->attr, 
                               (void*(*)(void*))&impl::thread_start, pimpl),
                "pthread_create");

    m_started = true;
}

void thread::join()
{
    if(is_started())
    {
        check_error(pthread_join(pimpl->thread,NULL), "pthread_join");
        if(!pimpl->error.empty())
            throw std::runtime_error(pimpl->error);

        m_started = false;
    }
}

bool thread::try_join(double wait_secs)
{
    if(is_started())
    {
        if(wait_secs > 0)
        {
            timespec ts;
            if(clock_gettime(CLOCK_REALTIME, &ts) == -1)
            {
                check_error(errno,"clock_gettime");
                assert(false);
            }

            ts.tv_sec += (int)wait_secs;
            ts.tv_nsec += (wait_secs - (int)wait_secs)*1e9;

            int ret = pthread_timedjoin_np(pimpl->thread,NULL,&ts);
            if(ret == ETIMEDOUT)
                return false;

            check_error(ret,"pthread_timedjoin_np");
        }
        else
        {
            int ret = pthread_tryjoin_np(pimpl->thread, NULL);
            if(ret == ETIMEDOUT)
                return false;
            check_error(ret,"pthread_tryjoin_np");
        }

        if(!pimpl->error.empty())
            throw std::runtime_error(pimpl->error);

        m_started = false;
    }
    return true;
}

} // namespace rod
