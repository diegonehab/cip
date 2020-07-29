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
