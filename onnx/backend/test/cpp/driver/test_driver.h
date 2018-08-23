#include <iostream>
#include <queue>
#include <string>
#include <cstdlib>
#include <cstdio>

#include "onnx/onnx_pb.h"
#include "onnx/proto_utils.h"
#include "onnx/onnxifi_loader.h"
#include "onnxifi.h"

#include "gtest/gtest.h"

struct TestData{
	std::vector<std::string> input_filenames_;
	std::vector<std::string> output_filenames_;

	TestData(){}

	TestData(const std::vector<std::string> &input_filenames, const std::vector<std::string> &output_filenames){
		input_filenames_ = input_filenames;
		output_filenames_ = output_filenames;
	}
};
struct TestCase{
	TestCase(
		const std::string& model_filename,
		const std::string& model_dirname,
		const std::vector<TestData>& test_data){

		model_filename_ = model_filename;
		model_dirname_ = model_dirname;
		test_data_ = test_data;
	}

	TestCase(){}

	std::string model_filename_;
	std::string model_dirname_;
	std::vector<TestData> test_data_;
};

struct ProtoTestData{
	std::vector<std::string> raw_inputs_;
	std::vector<std::string> raw_outputs_;
	std::vector<onnx::TensorProto> inputs_;
	std::vector<onnx::TensorProto> outputs_;
};

struct ProtoTestCase{
	std::string raw_model_;
	onnx::ModelProto model_;
	std::vector<ProtoTestData> proto_test_data_;
};

class TestDriver{

	std::string default_dir_;

	void setDefaultDir(const std::string& s);
	public:
	std::vector<TestCase> testcases_;
	TestDriver(const std::string default_dir = "."){
		default_dir_ = default_dir_;
	}

	int fetchAllTestCases(const std::string& _target_dir);
        int fetchSingleTestCase(const std::string& case_dir);
};

std::vector<TestCase> getTestCase();

void loadSingleFile(const std::string& filename, std::string& filedata);
ProtoTestCase loadSingleTestCase(const TestCase& t);
std::vector<ProtoTestCase> loadAllTestCases(const std::string& location);
std::vector<ProtoTestCase> loadAllTestCases(const std::vector<TestCase>& t);

//Need to be migrated to util in the future
onnxTensorDescriptorV1 ProtoToOnnxTensorDescriptor(const onnx::TensorProto& proto_tensor);
