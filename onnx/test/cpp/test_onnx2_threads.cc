// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem> // NOLINT(build/c++17)
#include <string>
#include <thread>
#include <vector>

#include "onnx/onnx2/cpu/common_helpers.h"
#include "onnx/onnx2/cpu/onnx2.h"
#include "onnx/onnx2/cpu/onnx2_helper.h"
#include "onnx/onnx2/cpu/thread_pool.h"

using namespace ONNX_NAMESPACE::v2;
using namespace ONNX_NAMESPACE::v2::utils;

TEST(onnx2_threads, CreateAndDestroy) {
  ThreadPool pool;
  pool.Start(4);
  EXPECT_EQ(pool.GetThreadCount(), 4);
}

TEST(onnx2_threads, SubmitSingleTask) {
  ThreadPool pool;
  pool.Start(2);
  int result = 0;
  auto task = [&result]() {
    for (size_t i = 0; i < 42; ++i) {
      result += 1;
    }
  };
  pool.SubmitTask(task);
  pool.Wait();
  EXPECT_EQ(result, 42);
}

TEST(onnx2_threads, SubmitMultipleTasks) {
  ThreadPool pool;
  pool.Start(4);
  constexpr int num_tasks = 100;
  std::atomic<int> counter(0);
  for (int i = 0; i < num_tasks; ++i) {
    pool.SubmitTask([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
  }
  pool.Wait();
  EXPECT_EQ(counter.load(), num_tasks);
}

TEST(onnx2_threads, ParallelExecution) {
  ThreadPool pool;
  pool.Start(8);

  std::atomic<int> counter(0);
  std::vector<int> thread_ids;
  std::mutex mutex;

  constexpr int num_tasks = 20;
  for (int i = 0; i < num_tasks; ++i) {
    pool.SubmitTask([&counter, &thread_ids, &mutex]() {
      int thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
      {
        std::lock_guard<std::mutex> lock(mutex);
        thread_ids.push_back(thread_id);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      counter.fetch_add(1, std::memory_order_relaxed);
    });
  }

  pool.Wait();

  EXPECT_EQ(counter.load(), num_tasks);

  std::sort(thread_ids.begin(), thread_ids.end());
  auto unique_end = std::unique(thread_ids.begin(), thread_ids.end());
  int unique_threads = std::distance(thread_ids.begin(), unique_end);
  EXPECT_GT(unique_threads, 1);
}

TEST(onnx2_threads, ParallelModelProcessing0) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("test_parallel_model");

  auto& graph = model.add_graph();

  const int num_tensors = 16;
  for (int i = 0; i < num_tensors; ++i) {
    auto& tensor = graph.add_initializer();
    std::vector<uint8_t> values(40, static_cast<uint8_t>(i));
    tensor.add_dims(1);
    tensor.add_dims(10);
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.set_raw_data(values);
  }

  // writing
  std::string temp_filename = "test_file_write_model_proto_parallel.onnx";
  {
    FileWriteStream stream(temp_filename);
    SerializeOptions options;
    model.SerializeToStream(stream, options);
  }

  // reading
  {
    FileStream stream(temp_filename);
    ParseOptions options;
    options.parallel = true;
    options.num_threads = 0;
    ModelProto model_proto2;
    stream.StartThreadPool(0);
    model_proto2.ParseFromStream(stream, options);
    stream.WaitForDelayedBlock();
    EXPECT_EQ(model_proto2.ref_ir_version(), model.ref_ir_version());
    EXPECT_EQ(model.ref_graph().ref_initializer().size(), model_proto2.ref_graph().ref_initializer().size());
  }

  std::remove(temp_filename.c_str());
}

TEST(onnx2_threads, ParallelModelProcessing4_File) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("test_parallel_model");

  auto& graph = model.add_graph();

  const int num_tensors = 16;
  for (int i = 0; i < num_tensors; ++i) {
    auto& tensor = graph.add_initializer();
    std::vector<uint8_t> values(40, static_cast<uint8_t>(i));
    tensor.add_dims(1);
    tensor.add_dims(10);
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.set_raw_data(values);
  }

  // writing
  std::string temp_filename = "test_file_write_model_proto_parallel.onnx";
  {
    FileWriteStream stream(temp_filename);
    SerializeOptions options;
    model.SerializeToStream(stream, options);
  }

  // reading
  {
    FileStream stream(temp_filename);
    ParseOptions options;
    options.parallel = true;
    options.num_threads = 2;
    ModelProto model_proto2;
    stream.StartThreadPool(2);
    model_proto2.ParseFromStream(stream, options);
    stream.WaitForDelayedBlock();
    EXPECT_EQ(model_proto2.ref_ir_version(), model.ref_ir_version());
    EXPECT_EQ(model.ref_graph().ref_initializer().size(), model_proto2.ref_graph().ref_initializer().size());
  }

  std::remove(temp_filename.c_str());
}

TEST(onnx2_threads, ParallelModelProcessing4_String) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("test_parallel_model");

  auto& graph = model.add_graph();

  const int num_tensors = 16;
  for (int i = 0; i < num_tensors; ++i) {
    auto& tensor = graph.add_initializer();
    std::vector<uint8_t> values(40, static_cast<uint8_t>(i));
    tensor.add_dims(1);
    tensor.add_dims(10);
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.set_raw_data(values);
  }

  // writing
  std::string serialized;
  {
    SerializeOptions options;
    model.SerializeToString(serialized, options);
  }

  // reading
  {
    ParseOptions options;
    options.parallel = true;
    options.num_threads = 2;
    ModelProto model_proto2;
    model_proto2.ParseFromString(serialized, options);
    EXPECT_EQ(model_proto2.ref_ir_version(), model.ref_ir_version());
    EXPECT_EQ(model.ref_graph().ref_initializer().size(), model_proto2.ref_graph().ref_initializer().size());
    for (size_t i = 0; i < model.ref_graph().ref_initializer().size(); ++i) {
      const auto& tensor1 = model.ref_graph().ref_initializer()[i];
      const auto& tensor2 = model_proto2.ref_graph().ref_initializer()[i];
      EXPECT_EQ(tensor1.ref_raw_data(), tensor2.ref_raw_data());
      EXPECT_EQ(tensor1.ref_data_type(), tensor2.ref_data_type());
    }
  }
}

TEST(onnx2_threads, ParallelModelProcessing4_FileExternalData) {
  ModelProto model;

  GraphProto& graph = model.add_graph();
  graph.set_name("test_graph");

  TensorProto& weights = graph.add_initializer();
  weights.set_name("weights");
  weights.set_data_type(TensorProto::DataType::FLOAT);
  weights.ref_dims().push_back(1);
  weights.ref_dims().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);
  weights.ref_raw_data().push_back(1);

  NodeProto& node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Add");
  AttributeProto& attr = node.add_attribute();
  attr.set_name("bias");
  TensorProto& biasw = attr.ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);
  biasw.ref_raw_data().push_back(2);

  NodeProto& nodeg = graph.add_node();
  nodeg.set_name("test_graph");
  nodeg.set_op_type("If");
  AttributeProto& attrg = nodeg.add_attribute();
  attrg.set_name("bias");
  GraphProto& nested = attrg.add_g();

  TensorProto& weights2 = nested.add_initializer();
  weights2.set_name("weights2");
  weights2.set_data_type(TensorProto::DataType::FLOAT);
  weights2.ref_dims().push_back(1);
  weights2.ref_dims().push_back(1);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);
  weights2.ref_raw_data().push_back(3);

  NodeProto& node2 = nested.add_node();
  node2.set_name("test_node");
  node2.set_op_type("Add");
  AttributeProto& attr2 = node2.add_attribute();
  attr2.set_name("bias");
  TensorProto& biasw2 = attr2.ref_t();
  biasw.set_name("biasw2");
  biasw2.set_data_type(TensorProto::DataType::FLOAT);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);
  biasw2.ref_raw_data().push_back(4);

  std::string temp_filename = "test_tensor_file_stream_read.tmp";
  std::string temp_weights = "test_tensor_file_stream_read.weight.tmp";

  {
    utils::TwoFilesWriteStream wstream(temp_filename, temp_weights);
    SerializeOptions wopts;
    wopts.raw_data_threshold = 2;
    SerializeProtoToStream(model, wstream, wopts);
  }

  ModelProto model2;
  {
    ParseOptions options;
    options.parallel = true;
    options.num_threads = 2;
    utils::TwoFilesStream rstream(temp_filename, temp_weights);
    rstream.StartThreadPool(2);
    ParseProtoFromStream(model2, rstream, options);
    rstream.WaitForDelayedBlock();
  }

  EXPECT_EQ(model.ref_graph().ref_initializer().size(), model2.ref_graph().ref_initializer().size());
  for (size_t i = 0; i < model.ref_graph().ref_initializer().size(); ++i) {
    EXPECT_EQ(
        model.ref_graph().ref_initializer()[i].ref_raw_data(), model2.ref_graph().ref_initializer()[i].ref_raw_data());
    EXPECT_EQ(
        model.ref_graph().ref_initializer()[i].ref_name().as_string(),
        model2.ref_graph().ref_initializer()[i].ref_name().as_string());
  }

  std::remove(temp_filename.c_str());
  std::remove(temp_weights.c_str());
}

TEST(onnx2_threads, ParallelModelProcessing4_FileExternalDataManyInitializers) {
  ModelProto model;

  GraphProto& graph = model.add_graph();
  graph.set_name("test_graph");

  for (uint8_t i = 0; i < 100; ++i) {
    TensorProto& weights = graph.add_initializer();
    weights.set_name("weights");
    weights.set_data_type(TensorProto::DataType::FLOAT);
    weights.ref_dims().push_back(1);
    weights.ref_dims().push_back(1);
    weights.ref_raw_data().push_back(i);
    weights.ref_raw_data().push_back(i);
    weights.ref_raw_data().push_back(i);
    weights.ref_raw_data().push_back(i);
  }

  NodeProto& node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Add");
  AttributeProto& attr = node.add_attribute();
  attr.set_name("bias");
  TensorProto& biasw = attr.ref_t();
  biasw.set_name("biasw");
  biasw.set_data_type(TensorProto::DataType::FLOAT);
  biasw.ref_dims().push_back(1);
  biasw.ref_dims().push_back(1);
  biasw.ref_raw_data().push_back(232);
  biasw.ref_raw_data().push_back(232);
  biasw.ref_raw_data().push_back(232);
  biasw.ref_raw_data().push_back(232);

  NodeProto& nodeg = graph.add_node();
  nodeg.set_name("test_graph");
  nodeg.set_op_type("If");
  AttributeProto& attrg = nodeg.add_attribute();
  attrg.set_name("bias");
  GraphProto& nested = attrg.add_g();

  for (uint8_t i = 0; i < 100; ++i) {
    TensorProto& weights2 = nested.add_initializer();
    weights2.set_name("weights2");
    weights2.set_data_type(TensorProto::DataType::FLOAT);
    weights2.ref_dims().push_back(1);
    weights2.ref_dims().push_back(1);
    weights2.ref_raw_data().push_back(105 + i);
    weights2.ref_raw_data().push_back(105 + i);
    weights2.ref_raw_data().push_back(105 + i);
    weights2.ref_raw_data().push_back(105 + i);
  }

  NodeProto& node2 = nested.add_node();
  node2.set_name("test_node");
  node2.set_op_type("Add");
  AttributeProto& attr2 = node2.add_attribute();
  attr2.set_name("bias");
  TensorProto& biasw2 = attr2.ref_t();
  biasw.set_name("biasw2");
  biasw2.set_data_type(TensorProto::DataType::FLOAT);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_dims().push_back(1);
  biasw2.ref_raw_data().push_back(244);
  biasw2.ref_raw_data().push_back(244);
  biasw2.ref_raw_data().push_back(244);
  biasw2.ref_raw_data().push_back(244);

  std::string temp_filename = "test_tensor_file_stream_read.big.tmp";
  std::string temp_weights = "test_tensor_file_stream_read.weight.big.tmp";

  {
    utils::TwoFilesWriteStream wstream(temp_filename, temp_weights);
    SerializeOptions wopts;
    wopts.raw_data_threshold = 2;
    SerializeProtoToStream(model, wstream, wopts);
  }

  {
    utils::TwoFilesWriteStream wstream(temp_filename, temp_weights);
    SerializeOptions wopts;
    wopts.raw_data_threshold = 2;
    SerializeProtoToStream(model, wstream, wopts);
  }

  int64_t length;
  {
    std::ifstream file(temp_weights, std::ios::binary | std::ios::ate);
    length = static_cast<int64_t>(file.tellg());
  }
  EXT_ENFORCE(length, 100 + 2 * 4);
  {
    std::ifstream file(temp_weights, std::ios::binary);
    std::vector<uint8_t> buffer(length);
    file.read(reinterpret_cast<char*>(buffer.data()), length);
    for (size_t i = 0; i < buffer.size(); ++i) {
      if (i < 4)
        EXPECT_EQ(buffer[i], static_cast<uint8_t>(232)) << " at index " << i;
      else if (i < 8)
        EXPECT_EQ(buffer[i], static_cast<uint8_t>(244)) << " at index " << i;
      else if (i < 408)
        EXPECT_EQ(buffer[i], static_cast<uint8_t>((i - 8) / 4 + 105)) << " at index " << i;
      else
        EXPECT_EQ(buffer[i], static_cast<uint8_t>((i - 408) / 4)) << " at index " << i;
    }
  }

  ModelProto model2;
  {
    ParseOptions options;
    options.parallel = true;
    options.num_threads = 2;
    utils::TwoFilesStream rstream(temp_filename, temp_weights);
    rstream.StartThreadPool(2);
    ParseProtoFromStream(model2, rstream, options);
    rstream.WaitForDelayedBlock();
  }

  EXPECT_EQ(model.ref_graph().ref_initializer().size(), model2.ref_graph().ref_initializer().size());
  for (size_t i = 0; i < model.ref_graph().ref_initializer().size(); ++i) {
    EXPECT_EQ(
        model.ref_graph().ref_initializer()[i].ref_raw_data(), model2.ref_graph().ref_initializer()[i].ref_raw_data());
    EXPECT_EQ(
        model.ref_graph().ref_initializer()[i].ref_name().as_string(),
        model2.ref_graph().ref_initializer()[i].ref_name().as_string());
  }

  std::remove(temp_filename.c_str());
  std::remove(temp_weights.c_str());
}
