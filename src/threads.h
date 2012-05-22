#ifndef NLFILTER_THREADS_H
#define NLFILTER_THREADS_H

namespace rod
{

class mutex
{
public:
    mutex();
    ~mutex();

    void lock();
    void unlock();

private:
    friend class condition_variable;

    struct impl;
    impl *pimpl;
};

class unique_lock
{
public:
    unique_lock(mutex &mtx) : m_mtx(mtx)
    {
        m_mtx.lock();
        m_locked = true;
    }
    ~unique_lock()
    {
        if(m_locked)
            unlock();
    }

    void lock()
    {
        assert(!m_locked);
        m_mtx.lock();
        m_locked = true;
    }

    void unlock()
    {
        assert(m_locked);
        m_mtx.unlock();
        m_locked = false;
    }
private:
    mutex &m_mtx;
    bool m_locked;

    friend class condition_variable;
};

class condition_variable
{
public:
    condition_variable();
    ~condition_variable();

    void wait(unique_lock &lk);
    void signal();

private:
    struct impl;
    impl *pimpl;
};

class thread
{
public:
    typedef void (*function_type)(void *);

    thread(function_type thread_start, void *data, bool start);
    ~thread();

    void start();
    bool is_started() const { return m_started; }

    void join();

    // returns true if thread was joined, false otherwise
    bool try_join(double wait_secs=0);
private:
    struct impl;
    impl *pimpl;

    bool m_started;
};

}


#endif
