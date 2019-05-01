#include <iostream>
#include <queue>
#include <cstdlib>

#include "onnx/onnxifi_loader.h"
#include "onnx/onnxifi_utils.h"
#include "onnx/string_utils.h"

#include "gtest/gtest.h"
namespace ONNX_NAMESPACE {
namespace testing {
/**
 *	Store one chunk of test data files repos,
 *	includng input and output.
 *	No real data was loaded in this type of class,
 *	but only data location.
 *  Data loading is unsolved, this class CANNOT be treated as real data.
 */
struct UnsolvedTestData {
  std::vector<std::string> input_filenames_;
  std::vector<std::string> output_filenames_;
  UnsolvedTestData() {}
  UnsolvedTestData(
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
 *  Data loading is unsolved, this class CANNOT be treated as real data.
 */
struct UnsolvedTestCase {
  UnsolvedTestCase(
      const std::string& model_filename,
      const std::string& model_dirname,
      const std::vector<UnsolvedTestData>& test_data)
      : model_filename_(model_filename),
        model_dirname_(model_dirname),
        test_data_(test_data) {}

  UnsolvedTestCase() {}
  std::string test_case_name_;
  std::string model_filename_;
  std::string model_dirname_;
  std::vector<UnsolvedTestData> test_data_;
};

/**
 *	Store one chunk of test data,
 *	including raw input/output and protos.
 *	Real data was loaded in this type of class.
 *	Data loading is resolved, this class may contains large chunk of data.
 */
struct ResolvedTestData {
  std::vector<ONNX_NAMESPACE::TensorProto> inputs_;
  std::vector<ONNX_NAMESPACE::TensorProto> outputs_;
};

/**
 *	Store one test model,
 *	including raw model, model proto and several chunks of test data.
 *	Real data was loaded in this type of class.
 *	Data loading is resolved, this class may contains large chunk of data.
 */
struct ResolvedTestCase {
  ONNX_NAMESPACE::ModelProto model_;
  std::vector<ResolvedTestData> proto_test_data_;
  std::string test_case_name_;
};

/**
 *	Store all unloaded test cases in one repo.
 */
class TestDriver {
  std::string default_dir_;

 public:
  void SetDefaultDir(const std::string& s);
  std::vector<UnsolvedTestCase> testcases_;
  TestDriver(const std::string& default_dir = ".") {
    default_dir_ = default_dir_;
  }
  /**
   *	Fetch all test cases in target.
   *	Return true if success, and false if failed.
   */
  bool FetchAllTestCases(const std::string& target_dir);
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
   *	Regular file(s): output_X.pb, store one output tensor.
   */
  void FetchSingleTestCase(
      const std::string& case_dir,
      const std::string& test_case_name);
};

std::vector<UnsolvedTestCase> GetTestCase();

/**
 *	Load one proto file from filename as string to filedata.
 */
void LoadSingleFile(const std::string& filename, std::string& filedata);

/**
 *	Load one test case.
 */
ResolvedTestCase LoadSingleTestCase(const UnsolvedTestCase& t);

/**
 *	Load all test cases.
 */
std::vector<ResolvedTestCase> LoadAllTestCases(const std::string& location);
std::vector<ResolvedTestCase> LoadAllTestCases(
    const std::vector<UnsolvedTestCase>& t);

} // namespace testing
} // namespace ONNX_NAMESPACE
