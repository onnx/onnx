#include <ifstream>

#include "onnx/onnxifi_loader.h"
#include "test_driver.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <errno.h>
#endif

#define ERRNO_DIR_END 0

bool fileExist(const std::string& filename){
	#ifdef _WIN32
	return std::filesystem::exists(filename);
	#else
	return std::experimental::filesystem::exists(filename);
	#endif
}

bool TestDriver::setDefaultDir(const std::string &s){
	_default_dir = s;
}

//load single test case in case_dir to _testcases
int TestDriver::fetchSingleTestCase(char* case_dir){
	std::string model_name = str(case_dir) + "model.onnx";
	if (fileExist(model_name)){
		TestCase test_case;
		test_case.model_name = model_name;
		test_case.model_dir = str(case_dir);
		int case_count = 0;
		for (;;case_count++){

			std::vector<std::string> input_filenames, output_filenames;
			std::string input_name, output_name;
			std::string case_dirname = str(case_dir) + "/test_data_set_" + std::itoa(case_count)

			//input_name = data_dirname + "/input_" + std::itoa(data_count) + ".pb";
			output_name = data_dirname + "/output_" + "0" + ".pb";

			if (!fileExist(output_name)){
				break;
			}

			for (int data_count = 0; ; data_count++){
				input_name = data_dirname + "/input_" + std::itoa(data_count) + ".pb";
				output_name = data_dirname + "/output_" + std::itoa(data_count) + ".pb";
				if (!fileExist(input_name) && !fileExist(input_name)){
					break;
				}
				if (fileExist(input_name)){
					input_filenames.push_back(input_name);
				}
				if (fileExist(output_name)){
					output_filenames.push_back(output_name);
				}
			}
			TestData test_data(input_filenames, output_filenames);
			test_case.test_data_.push_back(test_data);
		}
		_testcases.push_back(test_case);
	}
	return 0;
}

//load all test data in target_dir to _testcases
int TestDriver::fetchAllTestCases(const char* target_dir){
	DIR* directory = opendir(target_dir);
	if (directory == NULL){
		fprintf(stderr, "Error: cannot open directory %s when loading test data: %s\n",
			target_dir, strerr(errno));
		return -1;
	}

	while (true){
		errno = ERRNO_DIR_END;
		struct dirent* entry = readdir(directory);
		if (entry == NULL){
			if (errno != ERRNO_DIR_END){
				fprintf(stderr, "Error: cannot read directory %s when loading test data: s\n",
					target_dir, strerr(errno));
				return -1;
			} else{
				break;
			}
		fetchSingleTestCase(entry.d_name);
	}

cleanup:
	if (directory != NULL){
		if (closedir(directory) != 0){
			fprintf(stderr, "Warning: failed to close directory %s when loading test data: %s\n",
				target_dir, strerr(errno));
			return -1;
		}
	}
	return 0;
}

std::vector<TestCase> getTestCase(const std::string& location){
	TestDriver test_driver(location);
	test_driver.fetchTestCases();
	return test_driver._testcases;
}

void loadSingleFile(const std::string& filename, std::string& filedata){
	std::ifstream f(filename, std::ifstream::in);
	if (f.is_open(){
			std::string line;
		getline(f, filedata);
		while (getline(f, line)){
			filedata += "\n" + line;
		}
	}
	f.close();
}

ProtoTestCase loadSingleTestCase(const TestCase& t){

	ProtoTestCase st;
	loadSingleFile(t.model_filename_, st.raw_model_);
	ParseProtoFromPyBytes(&st.model_, st.raw_model_);

	for (auto test_data : t.test_data_){
		ProtoTestData proto_test_data;

		for (auto input_file : test_data.input_filenames){
			std::string input_data = "";
			loadSingleFile(input_file, input_data);
			proto_test_data.raw_inputs_.push_back(input_data);
			TensorProto input_proto;
			ParseProtoFromPyBytes(&input_proto, input_data);
			proto_test_data.inputs_.pushback(input_proto);
		}
		for (auto output_file : test_data.output_filenames){
			std::string output_data = "";
			loadSingleFile(output_file, output_data);
			proto_test_data.raw_outputs_.push_back(output_data);
			TensorProto output_proto;
			ParseProtoFromPyBytes(&output_proto, out_data);
			proto_test_data.outputs_.pushback(output_proto);
		}
	}
	return st;
}

vector<ProtoTestCase> loadAllTestCases(const std::string& location){
	vector<TestCase> t = getTestCase(location);
	return loadAllTestCases(t);
}

vector<ProtoTestCase> loadAllTestCases(const vector<TestCase>& t){
	vector<ProtoTestCase> st;
	for (auto i : t){
		st.push_back(loadSingleTestCase(i));
	}
	return st;
}

//Need to be migrated to util in the future
onnxTensorDescriptorV1 ProtoToOnnxTensorDescriptor(const TensorProto& proto_tensor){
	onnxTensorDescriptorV1 onnx_tensor;
	onnx_tensor.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
	onnx_tensor.name = proto_tensor.name;
	onnx_tensor.dataType = proto_tensor.data_type;
	onnx_tensor.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
	onnx_tensor.dimension = proto_tensor.dims.size();
	onnx_tensor.shape = proto_tensor.dims.data();
	onnx_tensor.buffer = proto_tensor.raw_data.data();
	return onnx_tensor;
}
