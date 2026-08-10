#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace gsic {

// Minimal fork-join pool. parallel_for splits [0, n) into chunks that workers
// (and the calling thread) claim from a shared atomic counter, which balances
// load without any per-task allocation.
//
// Deliberately built on plain std::thread rather than std::jthread: jthread
// and stop_token still need a very recent libc++, and this pool is the one
// piece of the engine every platform depends on.
class ThreadPool {
public:
    static ThreadPool& instance();

    // Worker threads plus the calling thread.
    int worker_count() const { return int(workers_.size()) + 1; }

    // fn(begin, end) is called on subranges of [0, n). Blocks until done.
    // Not reentrant: never call this from inside another parallel_for.
    void parallel_for(std::int64_t n, std::int64_t grain,
                      const std::function<void(std::int64_t, std::int64_t)>& fn);

    ~ThreadPool();

private:
    ThreadPool();
    void worker_loop();

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;

    // Current job, published by bumping generation_ under mutex_.
    const std::function<void(std::int64_t, std::int64_t)>* job_ = nullptr;
    std::int64_t job_n_ = 0;
    std::int64_t job_grain_ = 1;
    std::uint64_t generation_ = 0;
    bool stopping_ = false;

    std::atomic<std::int64_t> next_{0};
    std::atomic<int> active_{0};
};

} // namespace gsic
