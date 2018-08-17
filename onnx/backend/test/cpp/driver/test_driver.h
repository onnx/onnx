#include <iostream>
#include <queue>
#include <string>
struct TestData{
	std::vector<std::string> input_filenames_;
	std::vector<std::string> output_filenames;

	TestData(){}

	TestData(const std::vector<std::string> &input_filenames, const std::vector<std::string> &output_filenames){
		input_filenames_ = input_filenames;
		output_filenames = output_filenames;
	}
}
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
	std::vector<std::TestData> test_data_;
};

struct ProtoTestData{
	std::vector<std::string> raw_inputs_;
	std::vector<std::string> raw_outputs_;
	std::vector<TensorProto> inputs_;
	std::vector<TensorProto> outputs_;
}

struct ProtoTestCase{
	std::string raw_model_;
	ModelProto model_;
	std::vector<ProtoTestData> proto_test_data_;
};

class TestDriver{

	std::vector<TestCase> testcases_;
	std::string default_dir_;

	bool setDefaultDir(const std::string& s);
	TestDriver(const std::string default_dir = "."){
		default_dir_ = default_dir_;
		testcases_.clear();
	}

	int fetchAllTestCases(const std::string& target_dir = default_dir_);
	int fetchSingleTestCase(char* case_dir);
};

vector<TestCase> getTestCase(const std::string& location);

void loadSingleFile(const std::string& filename, std::string& filedata);
ProtoTestCase loadSingleTestCase(const TestCase& t);
vector<ProtoTestCase> loadAllTestCases(const std::string& location);
vector<ProtoTestCase> loadAllTestCases(const std::vector<testCase& t);

//Need to be migrated to util in the future
onnxTensorDescriptorV1 ProtoToOnnxTensorDescriptor(TensorProto proto);
