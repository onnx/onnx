/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_driver.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <dirent.h>
#endif
#include "onnx/common/common.h"

namespace ONNX_NAMESPACE {
namespace testing {
bool FileExists(const std::string& filename) {
#ifdef _WIN32
  if (INVALID_FILE_ATTRIBUTES == GetFileAttributes(filename.c_str())) {
    return false;
  }
#else
  struct stat stats;
  if (lstat(filename.c_str(), &stats) != 0 || !S_ISREG(stats.st_mode)) {
    return false;
  }
#endif
  return true;
}
void TestDriver::SetDefaultDir(const std::string& s) {
  default_dir_ = s;
}

/**
 *	It is possible that case_dir is not a dir.
 *	But it does not affect the result.
 */
void TestDriver::FetchSingleTestCase(const std::string& case_dir, const std::string& test_case_name) {
  std::string model_name = case_dir;
  model_name += "model.onnx";
  if (FileExists(model_name)) {
    UnsolvedTestCase test_case;
    test_case.test_case_name_ = test_case_name;
    test_case.model_filename_ = model_name;
    test_case.model_dirname_ = case_dir;
    for (int case_count = 0;; case_count++) {
      std::vector<std::string> input_filenames, output_filenames;
      std::string input_name, output_name;
      std::string case_dirname = case_dir;
      case_dirname += "test_data_set_" + ONNX_NAMESPACE::to_string(case_count);
      input_name = case_dirname + "/input_" + "0" + ".pb";
      output_name = case_dirname + "/output_" + "0" + ".pb";
      if (!FileExists(output_name) && !FileExists(input_name)) {
        break;
      }

      for (int data_count = 0;; data_count++) {
        input_name = case_dirname + "/input_" + ONNX_NAMESPACE::to_string(data_count) + ".pb";
        output_name = case_dirname + "/output_" + ONNX_NAMESPACE::to_string(data_count) + ".pb";
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
      UnsolvedTestData test_data(input_filenames, output_filenames);
      test_case.test_data_.emplace_back(std::move(test_data));
    }
    testcases_.emplace_back(std::move(test_case));
  }
}

bool TestDriver::FetchAllTestCases(const std::string& target) {
  std::string target_dir = target;
  if (target_dir == "") {
    target_dir = default_dir_;
  }
  if (target_dir[target_dir.size() - 1] == '/') {
    target_dir.erase(target_dir.size() - 1, 1);
  }
#ifdef _WIN32
  _finddata_t file;
  intptr_t lf;
  if ((lf = _findfirst(target_dir.c_str(), &file)) == -1) {
    std::cerr << "Error: cannot open directory " << target_dir << " when fetching test data: " << strerror(errno)
              << std::endl;
    return false;
  } else {
    ONNX_TRY {
      do {
        std::string entry_dname = file.name;
        std::string full_entry_dname;
        if (entry_dname != "." && entry_dname != "..") {
          full_entry_dname = target_dir + "/" + entry_dname + "/";
          FetchSingleTestCase(full_entry_dname, entry_dname);
        }
      } while (_findnext(lf, &file) == 0);
    }
    ONNX_CATCH(const std::exception& e) {
      ONNX_HANDLE_EXCEPTION([&]() {
        _findclose(lf);
        ONNX_THROW_EX(e);
      });
    }
    _findclose(lf);
  }
#else
  DIR* directory;
  ONNX_TRY {
    directory = opendir(target_dir.c_str());
    if (directory == NULL) {
      std::cerr << "Error: cannot open directory " << target_dir << " when fetching test data: " << strerror(errno)
                << std::endl;
      return false;
    }
    while (true) {
      errno = 0;
      struct dirent* entry = readdir(directory);
      if (entry == NULL) {
        if (errno != 0) {
          std::cerr << "Error: cannot read directory " << target_dir << " when fetching test data: " << strerror(errno)
                    << std::endl;
          return -1;
        } else {
          break;
        }
      }
      std::string full_entry_dname;
      std::string entry_dname = entry->d_name;
      if (entry_dname != "." && entry_dname != "..") {
        full_entry_dname = target_dir + "/" + entry_dname + "/";
        FetchSingleTestCase(full_entry_dname, entry_dname);
      }
    }
  }
  ONNX_CATCH(const std::exception& e) {
    ONNX_HANDLE_EXCEPTION([&]() {
      if (directory != NULL) {
        closedir(directory);
      }
      ONNX_THROW_EX(e);
    });
  }
  if (directory != NULL) {
    if (closedir(directory) != 0) {
      std::cerr << "Warning: failed to close directory " << target_dir
                << " when fetching test data: " << strerror(errno) << std::endl;
      return false;
    }
  }
#endif
  return true;
}

std::vector<UnsolvedTestCase> GetTestCase(const std::string& location) {
  TestDriver test_driver;
  test_driver.FetchAllTestCases(location);
  return test_driver.testcases_;
}

void LoadSingleFile(const std::string& filename, std::string& filedata) {
  FILE* fp;
  if ((fp = fopen(filename.c_str(), "r")) != NULL) {
    ONNX_TRY {
      int fsize;
      char buff[1024] = {0};
      do {
        fsize = fread(buff, sizeof(char), 1024, fp);
        filedata += std::string(buff, buff + fsize);
      } while (fsize == 1024);
    }
    ONNX_CATCH(const std::exception& e) {
      ONNX_HANDLE_EXCEPTION([&]() {
        fclose(fp);
        ONNX_THROW_EX(e);
      });
    }
    fclose(fp);
  } else {
    std::cerr << "Warning: failed to open file: " << filename << std::endl;
  }
}

ResolvedTestCase LoadSingleTestCase(const UnsolvedTestCase& t) {
  ResolvedTestCase st;
  std::string raw_model;
  LoadSingleFile(t.model_filename_, raw_model);
  st.test_case_name_ = t.test_case_name_;
  ONNX_NAMESPACE::ParseProtoFromBytes(&st.model_, raw_model.c_str(), raw_model.size());
  int test_data_counter = 0;
  for (auto& test_data : t.test_data_) {
    ResolvedTestData proto_test_data;
    for (auto& input_file : test_data.input_filenames_) {
      std::string input_data;
      ONNX_NAMESPACE::ValueInfoProto input_info;
      LoadSingleFile(input_file, input_data);
      input_info = st.model_.graph().input(test_data_counter);
      if (input_info.type().has_tensor_type()) {
        ONNX_NAMESPACE::TensorProto input_proto;
        ONNX_NAMESPACE::ParseProtoFromBytes(&input_proto, input_data.c_str(), input_data.size());
        proto_test_data.inputs_.emplace_back(std::move(input_proto));
      } else if (input_info.type().has_sequence_type()) {
        ONNX_NAMESPACE::SequenceProto input_proto;
        ONNX_NAMESPACE::ParseProtoFromBytes(&input_proto, input_data.c_str(), input_data.size());
        proto_test_data.seq_inputs_.emplace_back(std::move(input_proto));
      } else if (input_info.type().has_map_type()) {
        ONNX_NAMESPACE::MapProto input_proto;
        ONNX_NAMESPACE::ParseProtoFromBytes(&input_proto, input_data.c_str(), input_data.size());
        proto_test_data.map_inputs_.emplace_back(std::move(input_proto));
      } else if (input_info.type().has_optional_type()) {
        ONNX_NAMESPACE::OptionalProto input_proto;
        ONNX_NAMESPACE::ParseProtoFromBytes(&input_proto, input_data.c_str(), input_data.size());
        proto_test_data.optional_inputs_.emplace_back(std::move(input_proto));
      }
      test_data_counter++;
    }
    test_data_counter = 0;
    for (auto& output_file : test_data.output_filenames_) {
      std::string output_data;
      ONNX_NAMESPACE::ValueInfoProto output_info;
      output_info = st.model_.graph().output(test_data_counter);
      LoadSingleFile(output_file, output_data);
      if (output_info.type().has_tensor_type()) {
        ONNX_NAMESPACE::TensorProto output_proto;
        ONNX_NAMESPACE::ParseProtoFromBytes(&output_proto, output_data.c_str(), output_data.size());
        proto_test_data.outputs_.emplace_back(std::move(output_proto));
      } else if (output_info.type().has_sequence_type()) {
        ONNX_NAMESPACE::SequenceProto output_proto;
        ONNX_NAMESPACE::ParseProtoFromBytes(&output_proto, output_data.c_str(), output_data.size());
        proto_test_data.seq_outputs_.emplace_back(std::move(output_proto));
      } else if (output_info.type().has_map_type()) {
        ONNX_NAMESPACE::MapProto output_proto;
        ONNX_NAMESPACE::ParseProtoFromBytes(&output_proto, output_data.c_str(), output_data.size());
        proto_test_data.map_outputs_.emplace_back(std::move(output_proto));
      } else if (output_info.type().has_optional_type()) {
        ONNX_NAMESPACE::OptionalProto output_proto;
        ONNX_NAMESPACE::ParseProtoFromBytes(&output_proto, output_data.c_str(), output_data.size());
        proto_test_data.optional_outputs_.emplace_back(std::move(output_proto));
      }
      test_data_counter++;
    }
    st.proto_test_data_.emplace_back(std::move(proto_test_data));
  }
  return st;
}

std::vector<ResolvedTestCase> LoadAllTestCases(const std::string& location) {
  std::vector<UnsolvedTestCase> t = GetTestCase(location);
  return LoadAllTestCases(t);
}

std::vector<ResolvedTestCase> LoadAllTestCases(const std::vector<UnsolvedTestCase>& t) {
  std::vector<ResolvedTestCase> st;
  for (const auto& i : t) {
    st.push_back(LoadSingleTestCase(i));
  }
  return st;
}

} // namespace testing
} // namespace ONNX_NAMESPACE
