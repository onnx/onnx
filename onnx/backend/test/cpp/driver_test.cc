#include <gtest/gtest.h>

#include "driver/test_driver.h"
#include "onnx/onnxifi_loader.h"
#include "onnx/onnxifi.h"

class ONNXCppUnitTest
	: public testing::TestWithParam<ProtoTestCase>{
  protected:
	  std::vector<std::string> Y_by_cal;
	  std::vector<std::string> X_;
	  std::vector<std::string> Y_;
	  std::string model_;
	  void SetUp override {
		ProtoTestCase t = GetParam();
	    X_ = t.inputs_;
		Y_ = t.outputs_;
		model_ = t.model_;
	  }
	  void RunAndVerify(onnxifi_library lib, onnxBackend backend){

		onnxGraph graph;
		uint32_t weightCount = X_.model_.graph.intializer_size();
		onnxTensorDescriptorV1 weightDescriptors =
			ProtoToOnnxTensorDescriptor(X_.model_.graph.initializer(0));

		EXPECT_EQ(lib.onnxInitGraph(
				backend,
				NULL,
				X_.model_.size(),
				X_.model_.data(),
				weightCount,
				weightDescriptors,
				&graph),
			ONNXIFI_STATUS_SUCCESS);

		for (auto proto_test_data : X_.proto_test_data_){

			vector<onnxTensorDescriptorV1> input_descriptor, output_descriptor, result_descriptor;
			for (auto input : proto_test_data.inputs){
				input_descriptor.push_back(ProtoToOnnxTensorDescriptor(input));
			}
			for (auto output : proto_test_data.outputs){
				output_descriptor.push_back(ProtoToOnnxTensorDescriptor(output));
				onnxTensorDescriptorV1 result;
				result.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
				result.name = "result";
				result.dataType = output.data_type;
				result.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
				result.dimension = output.dims.size();
				result.shape = output.dims.data();
				result.buffer = (onnxPointer) new char[sizeof(output.raw_data_size())]
				result_descriptor.push_back(result);
			}
			EXPECT_EQ(lib.onnxSetGraphIO(graph,
										input_descriptor.size(),
										input_descriptor.data(),
										result_descriptor.size(),
										result_descriptor.data()),
					ONNXIFI_STATUS_SUCCESS);

			onnxEvent inputFence, outputFence;
			inputFence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
			outputFence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
			inputFence.type = ONNXIFI_SYNCHRONIZATION_EVENT;
			outputFence.type = ONNXIFI_SYNCHRONIZATION_EVENT;

			EXPECT_EQ(lib.onnxInitEvent(backend, &inputFence), ONNXIFI_STATUS_SUCCESS);
			EXPECT_EQ(lib.onnxRunGraph(graph, &inputFence, &outputFence),
					ONNXIFI_STATUS_SUCCESS);
			EXPECT_EQ(onnxWaitEvent(outputFence.event), ONNXIFI_STATUS_SUCCESS);
			for (int i = 0; i < output_descriptor.size(); i++){
				for (int j = 0; j < output_descriptor[i].raw_data_size(); j++){
					EXPECT_EQ(output_descriptor[i].raw_data[j], result_descriptor[i].raw_data[j]);
				}
			}
		}
		EXPECT_EQ(lib.onnxReleaseGraph(graph), ONNXIFI_STATUS_SUCCESS);
	}
}
TEST_P(ONNXCppUnitTest, ONNXCppUnitTestDriver){
	onnxifi_library lib;
	EXPECT_TRUE(onnxifi_load(1, NULL, &lib));

	onnxBackendID backendID;
	onnxBackend backend;
	onnxGraph graph;

	size_t numBackends;
	lib.onnxGetBackendIDs(&backendID, &numBackends);
////////////////Might be a problem////////////////
	const uint64_t backendProperties[] = {
		ONNXIFI_BACKEND_PROPERTY_NONE;
	};
	lib.onnxInitBackend(backendID, backendProperties, &backend);
	RunAndVerify();
	lib.onnxReleaseBackend(backend);
	lib.onnxReleaseBackendID(backendID);
}

INSTANTIATE_TEST_CASE_P(
	ONNXCppTest,
	ONNXCppUnitTest,
	serializeAllTestCases("/******** Write Me **********/"));
