#include "test_driver.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <errno.h>
#endif

namespace onnx {
namespace testing {
bool FileExists(const std::string& filename) {
  struct stat stats;
  if (lstat(filename.c_str(), &stats) != 0 || !S_ISREG(stats.st_mode)) {
    return false;
  }
  return true;
}

void TestDriver::SetDefaultDir(const std::string& s) {
  default_dir_ = s;
}

//load single test case in case_dir to _testcases
int TestDriver::FetchSingleTestCase(const std::string& case_dir) {
  std::string model_name = case_dir;
  model_name += "model.onnx";
  if (FileExists(model_name)) {
    TestCase test_case;
    test_case.model_filename_ = model_name;
    test_case.model_dirname_ = case_dir;
    for (int case_count = 0;; case_count++) {
      std::vector<std::string> input_filenames, output_filenames;
      std::string input_name, output_name;
      std::string case_dirname = case_dir;
      case_dirname += "test_data_set_" + to_string(case_count);

      output_name = case_dirname + "/output_" + "0" + ".pb";
      if (!FileExists(output_name) && !FileExists(input_name)) {
        break;
      }

      for (int data_count = 0;; data_count++) {
        input_name = case_dirname + "/input_" + to_string(data_count) + ".pb";
        output_name = case_dirname + "/output_" + to_string(data_count) + ".pb";
        const bool input_exists = FileExists(input_name);
        const bool output_exists = FileExists(output_name);
        if (!output_exists && !input_exists) {
          break;
        }
        if (input_exists) {
          input_filenames.emplace_back(std::move(input_name));
        }
        if (output_exists) {
          output_filenames.emplace_back(std::move(output_name));
        }
      }
      TestData test_data(input_filenames, output_filenames);
      test_case.test_data_.emplace_back(std::move(test_data));
    }
    testcases_.emplace_back(std::move(test_case));
  }
  return 0;
}

//load all test data in target_dir to _testcases
int TestDriver::FetchAllTestCases(const std::string& target) {
  std::string target_dir = target;
  if (target_dir == "") {
    target_dir = default_dir_;
  }
  if (target_dir[target_dir.size() - 1] == '/') {
    target_dir.erase(target_dir.size() - 1, 1);
  }
  DIR* directory;
  try {
    directory = opendir(target_dir.c_str());
    if (directory == NULL) {
      std::cerr << "Error: cannot open directory " << target_dir
                << " when fetching test data: " << strerror(errno) << std::endl;
      return -1;
    }
    while (true) {
      errno = 0;
      struct dirent* entry = readdir(directory);
      if (entry == NULL) {
        if (errno != 0) {
          std::cerr << "Error: cannot read directory " << target_dir
                    << " when fetching test data: " << strerror(errno)
                    << std::endl;
          return -1;
        } else {
          break;
        }
      }
      std::string entry_dname = entry->d_name;
      entry_dname = target_dir + '/' + entry_dname + "/";
      FetchSingleTestCase(entry_dname);
    }
  } catch (const std::exception& e) {
    if (directory != NULL) {
      if (closedir(directory) != 0) {
        std::cerr << "Warning: failed to close directory " << target_dir
                  << " when fetching test data: " << strerror(errno)
                  << std::endl;
      }
    }
    std::cerr << "Error: exception occured: " << e.what() << std::endl;
    throw e;
  }
  if (directory != NULL) {
    if (closedir(directory) != 0) {
      std::cerr << "Warning: failed to close directory " << target_dir
                << " when fetching test data: " << strerror(errno) << std::endl;
      return -1;
    }
  }
  return 0;
}

std::vector<TestCase> GetTestCase(const std::string& location) {
  TestDriver test_driver;
  test_driver.FetchAllTestCases(location);
  return test_driver.testcases_;
}

// TODO: fix the reading by using faster fread
void LoadSingleFile(const std::string& filename, std::string& filedata) {
  FILE* fp;
  if ((fp = fopen(filename.c_str(), "r")) != NULL) {
    try {
      std::string buffer;
      while (!feof(fp)) {
        buffer += fgetc(fp);
      }
      filedata += buffer;
    } catch (const std::exception& e) {
      fclose(fp);
      throw e;
    }
    fclose(fp);
  } else {
    std::cerr << "Warning: failed to open file: " << filename << std::endl;
  }
}

ProtoTestCase LoadSingleTestCase(const TestCase& t) {
  ProtoTestCase st;
  std::string raw_model;
  LoadSingleFile(t.model_filename_, raw_model);
  ParseProtoFromBytes(&st.model_, raw_model.c_str(), raw_model.size());

  for (auto& test_data : t.test_data_) {
    ProtoTestData proto_test_data;

    for (auto& input_file : test_data.input_filenames_) {
      std::string input_data;
      LoadSingleFile(input_file, input_data);
      onnx::TensorProto input_proto;
      ParseProtoFromBytes(&input_proto, input_data.c_str(), input_data.size());
      proto_test_data.inputs_.emplace_back(std::move(input_proto));
    }
    for (auto& output_file : test_data.output_filenames_) {
      std::string output_data = "";
      LoadSingleFile(output_file, output_data);
      onnx::TensorProto output_proto;
      ParseProtoFromBytes(
          &output_proto, output_data.c_str(), output_data.size());
      proto_test_data.outputs_.emplace_back(std::move(output_proto));
    }
  }
  return st;
}

std::vector<ProtoTestCase> LoadAllTestCases(const std::string& location) {
  std::vector<TestCase> t = GetTestCase(location);
  return LoadAllTestCases(t);
}

std::vector<ProtoTestCase> LoadAllTestCases(const std::vector<TestCase>& t) {
  std::vector<ProtoTestCase> st;
  for (auto i : t) {
    st.push_back(LoadSingleTestCase(i));
  }
  return st;
}

} // namespace testing
} // namespace onnx
