#include "driver/gtest_utils.h"
#include "onnx/checker.h"
#include "onnx/onnxifi.h"
#include "onnx/onnxifi_loader.h"
#define ONNXIFI_BACKEND_USED false

class ONNXCppDriverTest
	: public testing::TestWithParam<ProtoTestCase>{
  protected:
	  std::vector<ProtoTestData> protos_;
	  onnx::ModelProto model_;
	  void SetUp() override {
		ProtoTestCase t = GetParam();
		protos_ = t.proto_test_data_;
		model_ = t.model_;
	  }
	  uint64_t getDescriptorSize(onnxTensorDescriptorV1 *t){
		uint64_t d_size = 1;
		if (t->dimensions == 0) return 0;
		for (int i = 0; i < t->dimensions; i++){
			d_size *= t->shape[i];
		}
		return d_size;
	  }
	  void RunAndVerify(onnxifi_library& lib, onnxBackend& backend){
            // Check Model
            onnx::checker::check_model(model_);
            // Check Input&Output Tensors
            onnx::checker::CheckerContext ctx;
            for (auto proto_test_data : protos_) {
              for (auto input : proto_test_data.inputs_) {
                onnx::checker::check_tensor(input, ctx);
              }
              for (auto output : proto_test_data.outputs_) {
                onnx::checker::check_tensor(output, ctx);
              }
            }
            /* TO DO:
             * This chunk of code is to test the performance of onnxifi backend.
             * Since we are not using a real backend, we should wait and not
             * enable these tests. */

            if (ONNXIFI_BACKEND_USED) {
              onnxGraph graph;
              uint32_t weightCount = model_.graph().initializer_size();
              onnxTensorDescriptorV1 weightDescriptors =
                  ProtoToOnnxTensorDescriptor(model_.graph().initializer(0));

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

              for (auto proto_test_data : protos_) {
                std::vector<onnxTensorDescriptorV1> input_descriptor,
                    output_descriptor, result_descriptor;
                for (auto input : proto_test_data.inputs_) {
                  input_descriptor.push_back(
                      ProtoToOnnxTensorDescriptor(input));
                }
                for (auto output : proto_test_data.outputs_) {
                  output_descriptor.push_back(
                      ProtoToOnnxTensorDescriptor(output));
                  onnxTensorDescriptorV1 result;
                  result.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
                  result.name = "result";
                  result.dataType = output.data_type();
                  result.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
                  result.dimensions = output.dims().size();
                  result.shape = (unsigned long long*)output.dims().data();
                  result.buffer =
                      (onnxPointer) new char[sizeof(output.raw_data().size())];
                  result_descriptor.push_back(result);
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

                // EXPECT_EQ(lib.onnxInitEvent(backend, inputFence),
                // ONNXIFI_STATUS_SUCCESS);
                EXPECT_EQ(
                    lib.onnxRunGraph(graph, &inputFence, &outputFence),
                    ONNXIFI_STATUS_SUCCESS);
                EXPECT_EQ(
                    lib.onnxWaitEvent(outputFence.event),
                    ONNXIFI_STATUS_SUCCESS);
                for (int i = 0; i < output_descriptor.size(); i++) {
                  auto output_size = getDescriptorSize(&output_descriptor[i]);
                  for (int j = 0; j < output_size; j++) {
                    // size might be a problem!
                    EXPECT_EQ(
                        ((char*)output_descriptor[i].buffer)[j],
                        ((char*)result_descriptor[i].buffer)[j]);
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
