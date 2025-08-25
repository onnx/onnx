// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "onnx/onnx2/cpu/common_helpers.h"

namespace ONNX_NAMESPACE {
namespace v2 {
namespace utils {

class ThreadPool {
 public:
  ThreadPool();
  ~ThreadPool();
  void Start(int32_t num_threads);
  void SubmitTask(std::function<void()> job);
  void Wait();
  inline size_t GetThreadCount() const {
    return workers.size();
  }
  inline bool IsStarted() const {
    return is_started;
  }
  void Clear();

 private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> jobs;

  std::mutex queue_mutex;
  std::condition_variable condition;
  std::atomic<bool> stop;
  bool is_started;

  void worker_thread();
};

} // namespace utils
} // namespace v2
} // namespace ONNX_NAMESPACE
