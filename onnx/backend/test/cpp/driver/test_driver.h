#include <iostream>
#include <queue>
#include <cstdlib>

#include "onnx/onnxifi_loader.h"
#include "utils.h"

#include "gtest/gtest.h"
namespace onnx {
namespace testing {
/**
 *	Store one chunk of test data files repos,
 *	includng input and output.
 *	No real data was loaded in this type of class,
 *	but only data location.
 */
struct TestData {
  std::vector<std::string> input_filenames_;
  std::vector<std::string> output_filenames_;
  TestData() {}
  TestData(
      const std::vector<std::string>& input_filenames,
      const std::vector<std::string>& output_filenames)
      : input_filenames_(input_filenames),
        output_filenames_(output_filenames) {}
};

/**
 *	Store one model file repos,
 *	including one model file and several chunks of test data
 *	No real data was loaded in this type of class,
 *	but only data location.
 */
struct TestCase {
  TestCase(
      const std::string& model_filename,
      const std::string& model_dirname,
      const std::vector<TestData>& test_data)
      : model_filename_(model_filename),
        model_dirname_(model_dirname),
        test_data_(test_data) {}

  TestCase() {}

  std::string model_filename_;
  std::string model_dirname_;
  std::vector<TestData> test_data_;
};

/**
 *	Store one chunk of test data,
 *	including raw input/output and protos
 *	Real data was loaded in this type of class..
 */
struct ProtoTestData {
  std::vector<onnx::TensorProto> inputs_;
  std::vector<onnx::TensorProto> outputs_;
};

/**
 *	Store one test model,
 *	including raw model, model proto and several chunks of test data.
 *	Real data was loaded in this type of class.
 */
struct ProtoTestCase {
  onnx::ModelProto model_;
  std::vector<ProtoTestData> proto_test_data_;
};

/**
 *	Store all unloaded test cases in one repo.
 */
class TestDriver {
  std::string default_dir_;

 public:
  void SetDefaultDir(const std::string& s);
  std::vector<TestCase> testcases_;
  TestDriver(const std::string& default_dir = ".") {
    default_dir_ = default_dir_;
  }

  int FetchAllTestCases(const std::string& target_dir);
  /**
   *	Fetch one test case from repo case_dir.
   *	The structure of case_dir should include following contents.
   *
   *	Regular file: model.onnx, store the protobuf of the model.
   *	Repo(s): test_data_set_X: store the input&output data.
   *
   *	Each repo in test_data_set_X should include following contents.
   *
   *	Regular file(s): input_X.pb, store one input tensor.
   *	Reuglar file(s): output_X.pb, stroe one output tensor.
   */
  int FetchSingleTestCase(const std::string& case_dir);
};

std::vector<TestCase> GetTestCase();

/**
 *	Load one proto file from filename as string to filedata.
 */
void LoadSingleFile(const std::string& filename, std::string& filedata);

/**
 *	Load one test case.
 */
ProtoTestCase LoadSingleTestCase(const TestCase& t);

/**
 *	Load all test cases.
 */
std::vector<ProtoTestCase> LoadAllTestCases(const std::string& location);
std::vector<ProtoTestCase> LoadAllTestCases(const std::vector<TestCase>& t);

} // namespace testing
} // namespace onnx
