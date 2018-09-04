#include "test_driver.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <errno.h>
#endif

namespace ONNX_NAMESPACE {
namespace testing {
bool FileExists(const std::string& filename) {
  struct stat stats;
#ifdef _WIN32
  if (INVALID_FILE_ATTRIBUTE == GetFileAttributes(filename.c_str())) {
    return false;
  }
#else
  if (lstat(filename.c_str(), &stats) != 0 || !S_ISREG(stats.st_mode)) {
    return false;
  }
#endif
  return true;
}

void TestDriver::SetDefaultDir(const std::string& s) {
  default_dir_ = s;
}

void TestDriver::FetchSingleTestCase(const std::string& case_dir) {
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
}

//load all test data in target_dir to _testcases
bool TestDriver::FetchAllTestCases(const std::string& target) {
  std::string target_dir = target;
  if (target_dir == "") {
    target_dir = default_dir_;
  }
  if (target_dir[target_dir.size() - 1] == '/') {
    target_dir.erase(target_dir.size() - 1, 1);
  }
  // ifdef _WIN32
  //#else
  DIR* directory;
  try {
    directory = opendir(target_dir.c_str());
    if (directory == NULL) {
      std::cerr << "Error: cannot open directory " << target_dir
                << " when fetching test data: " << strerror(errno) << std::endl;
      return false;
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
    throw;
  }
  if (directory != NULL) {
    if (closedir(directory) != 0) {
      std::cerr << "Warning: failed to close directory " << target_dir
                << " when fetching test data: " << strerror(errno) << std::endl;
      return false;
    }
  }
  //#endif
  return true;
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
      int fsize;
      char buff[1024] = {0};
      do {
        fsize = fread(buff, sizeof(char), 1024, fp);
        filedata += std::string(buff, buff + fsize);
      } while (fsize == 1024);
    } catch (const std::exception& e) {
      fclose(fp);
      throw;
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
  ONNX_NAMESPACE::ParseProtoFromBytes(
      &st.model_, raw_model.c_str(), raw_model.size());

  for (auto& test_data : t.test_data_) {
    ProtoTestData proto_test_data;

    for (auto& input_file : test_data.input_filenames_) {
      std::string input_data;
      LoadSingleFile(input_file, input_data);
      ONNX_NAMESPACE::TensorProto input_proto;
      ONNX_NAMESPACE::ParseProtoFromBytes(
          &input_proto, input_data.c_str(), input_data.size());
      proto_test_data.inputs_.emplace_back(std::move(input_proto));
    }
    for (auto& output_file : test_data.output_filenames_) {
      std::string output_data = "";
      LoadSingleFile(output_file, output_data);
      ONNX_NAMESPACE::TensorProto output_proto;
      ONNX_NAMESPACE::ParseProtoFromBytes(
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
  for (const auto& i : t) {
    st.push_back(LoadSingleTestCase(i));
  }
  return st;
}

} // namespace testing
} // namespace ONNX_NAMESPACE
