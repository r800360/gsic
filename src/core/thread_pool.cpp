#include "thread_pool.h"

#include <algorithm>

namespace gsic {

ThreadPool& ThreadPool::instance() {
    static ThreadPool pool;
    return pool;
}

ThreadPool::ThreadPool() {
    const int n = int(std::max(1u, std::thread::hardware_concurrency())) - 1;
    workers_.reserve(n);
    for (int i = 0; i < n; ++i) workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
    // The stop flag is part of the wait predicate, so it must change while
    // the mutex is held; otherwise a worker between "predicate false" and
    // "blocked in wait" would miss the notify and never join.
    {
        std::lock_guard lk(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
}

void ThreadPool::worker_loop() {
    std::uint64_t seen = 0;
    while (true) {
        const std::function<void(std::int64_t, std::int64_t)>* job;
        std::int64_t n, grain;
        {
            std::unique_lock lk(mutex_);
            cv_.wait(lk, [&] { return stopping_ || generation_ != seen; });
            if (stopping_) return;
            seen = generation_;
            job = job_;
            n = job_n_;
            grain = job_grain_;
        }
        while (true) {
            const std::int64_t begin = next_.fetch_add(grain, std::memory_order_relaxed);
            if (begin >= n) break;
            (*job)(begin, std::min(begin + grain, n));
        }
        // The last worker to finish wakes the thread waiting in parallel_for.
        if (active_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard lk(mutex_);
            done_cv_.notify_all();
        }
    }
}

void ThreadPool::parallel_for(std::int64_t n, std::int64_t grain,
                              const std::function<void(std::int64_t, std::int64_t)>& fn) {
    if (n <= 0) return;
    grain = std::max<std::int64_t>(1, grain);
    if (workers_.empty() || n <= grain) {
        fn(0, n);
        return;
    }
    {
        std::lock_guard lk(mutex_);
        job_ = &fn;
        job_n_ = n;
        job_grain_ = grain;
        next_.store(0, std::memory_order_relaxed);
        active_.store(int(workers_.size()), std::memory_order_relaxed);
        ++generation_;
    }
    cv_.notify_all();
    // The calling thread joins the work instead of idling.
    while (true) {
        const std::int64_t begin = next_.fetch_add(grain, std::memory_order_relaxed);
        if (begin >= n) break;
        fn(begin, std::min(begin + grain, n));
    }
    // Workers only touch `fn` before decrementing active_, so waiting for
    // zero here guarantees no worker still holds the caller's lambda.
    std::unique_lock lk(mutex_);
    done_cv_.wait(lk, [&] { return active_.load(std::memory_order_acquire) == 0; });
}

} // namespace gsic
