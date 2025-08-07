#include "thread_pool.h"

namespace onnx2 {
namespace utils {

ThreadPool::ThreadPool() { is_started = false; }

void ThreadPool::Start(int32_t num_threads) {
  EXT_ENFORCE(workers.size() == 0, "ThreadPool already started");
  stop = false;
  is_started = true;
  if (num_threads == -1)
    num_threads = std::thread::hardware_concurrency();

  for (size_t i = 0; i < num_threads; ++i) {
    workers.emplace_back(&ThreadPool::worker_thread, this);
  }
}

void ThreadPool::SubmitTask(std::function<void()> job) {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    jobs.push(std::move(job));
  }
  if (!workers.empty())
    condition.notify_one();
}

void ThreadPool::worker_thread() {
  while (true) {
    std::function<void()> job;

    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      condition.wait(lock, [this]() { return stop || !jobs.empty(); });

      if (jobs.empty()) {
        if (stop)
          return;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      job = std::move(jobs.front());
      jobs.pop();
    }

    job();
  }
}

void ThreadPool::Wait() {
  if (workers.empty()) {
    // No workers so we manually run the jobs.
    while (!jobs.empty()) {
      std::function<void()> job = std::move(jobs.front());
      jobs.pop();
      job();
    }
  }
  while (jobs.size() > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  stop = true;
  if (!workers.empty()) {
    condition.notify_all();
  }
  for (std::thread &worker : workers) {
    if (worker.joinable())
      worker.join();
  }
  workers.clear();
  is_started = false;
}

ThreadPool::~ThreadPool() { Wait(); }

void ThreadPool::Clear() {
  EXT_ENFORCE(!IsStarted(), "Cannot clear the pool if threads are still running.");
  workers.clear();
  while (!jobs.empty()) {
    jobs.pop();
  }
}

} // namespace utils
} // namespace onnx2
