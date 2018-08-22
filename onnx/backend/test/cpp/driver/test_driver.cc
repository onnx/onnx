#include <fstream>
#include "test_driver.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <errno.h>
#endif

#define ERRNO_DIR_END 0

bool fileExist(const std::string& filename){
	FILE *fp;
	if ((fp = fopen(filename.c_str(), "r")) != NULL){
		fclose(fp);
		return true;
	}
	return false;
}

void TestDriver::setDefaultDir(const std::string &s){
	default_dir_ = s;
}

//load single test case in case_dir to _testcases
int TestDriver::fetchSingleTestCase(const std::string& case_dir) {
  std::string model_name = case_dir;
  model_name += "model.onnx";
  if (fileExist(model_name)) {
    TestCase test_case;
    test_case.model_filename_ = model_name;
    test_case.model_dirname_ = case_dir;
    int case_count = 0;
    for (;; case_count++) {
      std::vector<std::string> input_filenames, output_filenames;
      std::string input_name, output_name;
      std::string case_dirname = case_dir;
      case_dirname += "test_data_set_" + std::to_string(case_count);

      output_name = case_dirname + "/output_" + "0" + ".pb";
      if (!fileExist(output_name)) {
        break;
      }

      for (int data_count = 0;; data_count++) {
        input_name =
            case_dirname + "/input_" + std::to_string(data_count) + ".pb";
        output_name =
            case_dirname + "/output_" + std::to_string(data_count) + ".pb";
        if (!fileExist(input_name) && !fileExist(input_name)) {
          break;
        }
        if (fileExist(input_name)) {
          input_filenames.push_back(input_name);
        }
        if (fileExist(output_name)) {
          output_filenames.push_back(output_name);
        }
      }
      TestData test_data(input_filenames, output_filenames);
      test_case.test_data_.push_back(test_data);
    }
    testcases_.push_back(test_case);
  }
  return 0;
}

//load all test data in target_dir to _testcases
int TestDriver::fetchAllTestCases(const std::string& _target_dir){
	std::string target_dir = _target_dir;
	if (target_dir == ""){
		target_dir = default_dir_;
	}
	DIR* directory = opendir(target_dir.c_str());
	if (directory == NULL){
		fprintf(stderr, "Error: cannot open directory %s when loading test data: %s\n",
			target_dir.c_str(), strerror(errno));
		return -1;
	}

	while (true){
		errno = ERRNO_DIR_END;
		struct dirent* entry = readdir(directory);
		if (entry == NULL){
			if (errno != ERRNO_DIR_END){
				fprintf(stderr, "Error: cannot read directory %s when loading test data: %s\n",
					target_dir.c_str(), strerror(errno));
				return -1;
			} else{
				break;
			}
		}
                std::string entry_dname = entry->d_name;
                entry_dname = _target_dir + '/' + entry_dname + "/";
                fetchSingleTestCase(entry_dname);
        }
cleanup:
	if (directory != NULL){
		if (closedir(directory) != 0){
			fprintf(stderr, "Warning: failed to close directory %s when loading test data: %s\n",
				target_dir.c_str(), strerror(errno));
			return -1;
		}
	}
	return 0;
}

std::vector<TestCase> getTestCase(const std::string& location){
	TestDriver test_driver;
	test_driver.fetchAllTestCases(location);
	return test_driver.testcases_;
}

void loadSingleFile(const std::string& filename, std::string& filedata){
	std::ifstream f(filename, std::ifstream::in);
	if (f.is_open()){
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
	ParseProtoFromBytes(&st.model_, st.raw_model_.c_str(), st.raw_model_.size());

	for (auto test_data : t.test_data_){
		ProtoTestData proto_test_data;

		for (auto input_file : test_data.input_filenames_){
			std::string input_data = "";
			loadSingleFile(input_file, input_data);
			proto_test_data.raw_inputs_.push_back(input_data);
			onnx::TensorProto input_proto;
			ParseProtoFromBytes(&input_proto, input_data.c_str(), input_data.size());
			proto_test_data.inputs_.push_back(input_proto);
		}
		for (auto output_file : test_data.output_filenames_){
			std::string output_data = "";
			loadSingleFile(output_file, output_data);
			proto_test_data.raw_outputs_.push_back(output_data);
			onnx::TensorProto output_proto;
			ParseProtoFromBytes(&output_proto, output_data.c_str(), output_data.size());
			proto_test_data.outputs_.push_back(output_proto);
		}
	}
	return st;
}

std::vector<ProtoTestCase> loadAllTestCases(const std::string& location){
	std::vector<TestCase> t = getTestCase(location);
	return loadAllTestCases(t);
}

std::vector<ProtoTestCase> loadAllTestCases(const std::vector<TestCase>& t){
	std::vector<ProtoTestCase> st;
	for (auto i : t){
		st.push_back(loadSingleTestCase(i));
	}
	return st;
}

//Need to be migrated to util in the future
onnxTensorDescriptorV1 ProtoToOnnxTensorDescriptor(const onnx::TensorProto& proto_tensor){
	onnxTensorDescriptorV1 onnx_tensor;
	onnx_tensor.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
	onnx_tensor.name = proto_tensor.name().c_str();
	onnx_tensor.dataType = proto_tensor.data_type();
	onnx_tensor.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
	onnx_tensor.dimensions = proto_tensor.dims().size();
	onnx_tensor.shape = (uint64_t *)proto_tensor.dims().data();
	onnx_tensor.buffer = (onnxPointer)proto_tensor.raw_data().data();
	return onnx_tensor;
}
