#include "driver/gtest_utils.h"

#include "onnx/checker.h"
#include "onnx/onnxifi.h"
#include "onnx/onnxifi_loader.h"

const bool ONNXIFI_BACKEND_USED = false;
const float ONNXIFI_TESTDATA_EPS = 1e-5;

template <typename T>
class CompareOnnxifiData {
 public:
  bool IsEqual(void* P, void* Q) {
    T x = *((T*)P), y = *((T*)Q);
    return ((x - y) > -ONNXIFI_TESTDATA_EPS) &&
        ((x - y) < ONNXIFI_TESTDATA_EPS);
  }
};
class ONNXCppDriverTest
    : public testing::TestWithParam<onnx::testing::ProtoTestCase> {
 protected:
  std::vector<onnx::testing::ProtoTestData> protos_;
  ONNX_NAMESPACE::ModelProto model_;
  void SetUp() override {
    onnx::testing::ProtoTestCase t = GetParam();
    protos_ = t.proto_test_data_;
    model_ = t.model_;
  }
  uint64_t GetDescriptorSize(const onnxTensorDescriptorV1* t) {
    uint64_t d_size = 1;
    for (int i = 0; i < t->dimensions; i++) {
      d_size *= t->shape[i];
    }
    return d_size;
  }
  bool IsDescriptorEqual(
      const onnxTensorDescriptorV1& x,
      const onnxTensorDescriptorV1& y) {
    if (x.dataType != y.dataType || x.dimensions != y.dimensions) {
      return false;
    }
    const int dims = x.dimensions;
    for (int i = 0; i < dims; i++) {
      if (x.shape[i] != y.shape[i]) {
        return false;
      }
    }
    const uint64_t d_size = GetDescriptorSize(&x);
    void* p1 = (void*)x.buffer;
    void* p2 = (void*)y.buffer;
    bool is_equal = true;
    for (uint64_t i = 0; i < d_size; i++) {
      int offset = 1;
      switch (x.dataType) {
        case ONNXIFI_DATATYPE_UNDEFINED:
        case ONNXIFI_DATATYPE_INT8:
          CompareOnnxifiData<char> compare_int8;
          is_equal &= compare_int8.IsEqual(p1, p2);
          offset = 1;
          break;
        case ONNXIFI_DATATYPE_UINT8:
          CompareOnnxifiData<unsigned char> compare_uint8;
          is_equal &= compare_uint8.IsEqual(p1, p2);
          offset = 1;
          break;
        case ONNXIFI_DATATYPE_FLOAT16:
          // no support now
          break;
        case ONNXIFI_DATATYPE_INT16:
          CompareOnnxifiData<short> compare_int16;
          is_equal &= compare_int16.IsEqual(p1, p2);
          offset = 2;
          break;
        case ONNXIFI_DATATYPE_UINT16:
          CompareOnnxifiData<unsigned short> compare_uint16;
          is_equal &= compare_uint16.IsEqual(p1, p2);
          offset = 2;
          break;
        case ONNXIFI_DATATYPE_FLOAT32:
          CompareOnnxifiData<float> compare_float32;
          is_equal &= compare_float32.IsEqual(p1, p2);
          offset = 4;
          break;
        case ONNXIFI_DATATYPE_INT32:
          CompareOnnxifiData<int> compare_int32;
          is_equal &= compare_int32.IsEqual(p1, p2);
          offset = 4;
          break;
        case ONNXIFI_DATATYPE_UINT32:
          CompareOnnxifiData<unsigned int> compare_uint32;
          is_equal &= compare_uint32.IsEqual(p1, p2);
          offset = 4;
          break;
        case ONNXIFI_DATATYPE_FLOAT64:
          CompareOnnxifiData<long double> compare_float64;
          is_equal &= compare_float64.IsEqual(p1, p2);
          offset = 8;
          break;
        case ONNXIFI_DATATYPE_INT64:
          CompareOnnxifiData<long long> compare_int64;
          is_equal &= compare_int64.IsEqual(p1, p2);
          offset = 8;
          break;
        case ONNXIFI_DATATYPE_UINT64:
          CompareOnnxifiData<unsigned long long> compare_uint64;
          is_equal &= compare_uint64.IsEqual(p1, p2);
          offset = 8;
          break;
        case ONNXIFI_DATATYPE_COMPLEX64:
        case ONNXIFI_DATATYPE_COMPLEX128:
          // no support now
          break;
      }
      p1 = (char*)p1 + offset;
      p2 = (char*)p2 + offset;
      if (!is_equal) {
        return false;
      }
    }
    return true;
  }
  void RunAndVerify(onnxifi_library& lib, onnxBackend& backend) {
    // Check Model
    ONNX_NAMESPACE::checker::check_model(model_);
    // Check Input&Output Tensors
    ONNX_NAMESPACE::checker::CheckerContext ctx;
    for (auto proto_test_data : protos_) {
      for (auto input : proto_test_data.inputs_) {
        ONNX_NAMESPACE::checker::check_tensor(input, ctx);
      }
      for (auto output : proto_test_data.outputs_) {
        ONNX_NAMESPACE::checker::check_tensor(output, ctx);
      }
    }
    /* TO DO:
     * This chunk of code is to test the correctness of onnxifi backend.
     * Since we are not using a real backend, we should wait and not
     * enable these tests. */

    if (ONNXIFI_BACKEND_USED) {
      onnxGraph graph;
      uint32_t weightCount = model_.graph().initializer_size();
      onnxTensorDescriptorV1 weightDescriptors =
          onnx::testing::ProtoToOnnxTensorDescriptor(
              model_.graph().initializer(0));

      EXPECT_EQ(
          lib.onnxInitGraph(
              backend,
              NULL,
              sizeof(model_),
              &model_,
              weightCount,
              &weightDescriptors,
              &graph),
          ONNXIFI_STATUS_SUCCESS);

      for (const auto& proto_test_data : protos_) {
        std::vector<onnxTensorDescriptorV1> input_descriptor, output_descriptor,
            result_descriptor;
        for (const auto& input : proto_test_data.inputs_) {
          input_descriptor.push_back(
              onnx::testing::ProtoToOnnxTensorDescriptor(input));
        }
        int output_count = 0;
        for (auto& output : proto_test_data.outputs_) {
          output_count++;
          output_descriptor.push_back(
              onnx::testing::ProtoToOnnxTensorDescriptor(output));
          onnxTensorDescriptorV1 result;
          result.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
          std::string name_string =
              "output_" + onnx::testing::to_string(output_count);
          result.name = name_string.c_str();
          result.dataType = output.data_type();
          result.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
          std::vector<uint64_t> shape_values(
              output.dims().begin(), output.dims().end());
          result.dimensions = shape_values.size();
          result.shape = shape_values.data();
          std::vector<uint8_t> raw_data(output.raw_data().size(), 0);
          result.buffer = (onnxPointer)raw_data.data();
          result_descriptor.emplace_back(std::move(result));
        }
        EXPECT_EQ(
            lib.onnxSetGraphIO(
                graph,
                input_descriptor.size(),
                input_descriptor.data(),
                result_descriptor.size(),
                result_descriptor.data()),
            ONNXIFI_STATUS_SUCCESS);

        onnxMemoryFenceV1 inputFence, outputFence;
        inputFence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
        outputFence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
        inputFence.type = ONNXIFI_SYNCHRONIZATION_EVENT;
        outputFence.type = ONNXIFI_SYNCHRONIZATION_EVENT;

        EXPECT_EQ(
            lib.onnxRunGraph(graph, &inputFence, &outputFence),
            ONNXIFI_STATUS_SUCCESS);
        EXPECT_EQ(lib.onnxWaitEvent(outputFence.event), ONNXIFI_STATUS_SUCCESS);
        for (int i = 0; i < output_descriptor.size(); i++) {
          auto output_size = GetDescriptorSize(&output_descriptor[i]);
          for (int j = 0; j < output_size; j++) {
            EXPECT_EQ(
                IsDescriptorEqual(output_descriptor[i], result_descriptor[i]),
                true);
            ;
          }
        }
      }
      EXPECT_EQ(lib.onnxReleaseGraph(graph), ONNXIFI_STATUS_SUCCESS);
    }
  }
};
TEST_P(ONNXCppDriverTest, ONNXCppDriverUnitTest){
	onnxifi_library lib;
	onnxBackendID backendID;
	onnxBackend backend;
        // using default onnxifi backend
        if (ONNXIFI_BACKEND_USED) {
          EXPECT_TRUE(onnxifi_load(1, NULL, &lib));

          size_t numBackends;
          lib.onnxGetBackendIDs(&backendID, &numBackends);
          const uint64_t backendProperties[] = {ONNXIFI_BACKEND_PROPERTY_NONE};
          lib.onnxInitBackend(backendID, backendProperties, &backend);
        }
        RunAndVerify(lib, backend);
        if (ONNXIFI_BACKEND_USED) {
          lib.onnxReleaseBackend(backend);
          lib.onnxReleaseBackendID(backendID);
        }
}
INSTANTIATE_TEST_CASE_P(
	ONNXCppAllTest,
	ONNXCppDriverTest,
	testing::ValuesIn(all_test_cases));
