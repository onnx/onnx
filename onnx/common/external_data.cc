// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/common/external_data.h"

#include <cerrno>
#include <limits>
#include <string_view>
#include <vector>

#include "onnx/checker.h"
#include "onnx/common/path.h"
#include "onnx/common/scoped_resource.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
// path.h undefines the SDK annotation to avoid conflicts with ONNX enums.
#ifndef OPTIONAL
#define OPTIONAL
#define ONNX_DEFINED_WINDOWS_OPTIONAL
#endif
#include <winternl.h>
#ifdef ONNX_DEFINED_WINDOWS_OPTIONAL
#undef OPTIONAL
#undef ONNX_DEFINED_WINDOWS_OPTIONAL
#endif
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// NOLINTBEGIN(bugprone-signed-bitwise) -- System APIs require signed flag bitmasks.

namespace ONNX_NAMESPACE::checker {
using ONNX_NAMESPACE::path_to_utf8;
using ONNX_NAMESPACE::ScopedFd;
using ONNX_NAMESPACE::utf8_to_path;
#ifdef _WIN32
using ONNX_NAMESPACE::ScopedHandle;
#endif

// Treat both separators as path boundaries for portable lexical validation.
template <typename Pred>
static bool any_path_component(const std::filesystem::path& normalized_path, Pred pred) {
  // A Windows drive prefix is not part of the first component: "C:..\\data.bin"
  // must be checked as "..\\data.bin", not as a component named "C:..".
  const std::string path_str = path_to_utf8(normalized_path.relative_path());
  std::string_view rest = path_str;
  while (true) {
    const auto end = rest.find_first_of("/\\");
    if (pred(rest.substr(0, end))) {
      return true;
    }
    if (end == std::string_view::npos) {
      return false;
    }
    rest.remove_prefix(end + 1);
  }
}

// True if a lexically normalized path still has a '..' component, so it points
// outside the directory it is relative to. Names that merely contain two dots,
// such as "model..weights.bin", do not count.
static bool has_parent_directory_component(const std::filesystem::path& normalized_path) {
  return any_path_component(normalized_path, [](std::string_view part) { return part == ".."; });
}

// True if a path has a component made only of dots and spaces, such as "..." or
// ".. ". Windows trims trailing dots and spaces from each component, so it reads
// such a component as the current or parent directory.
static bool has_dots_and_spaces_component(const std::filesystem::path& normalized_path) {
  return any_path_component(normalized_path, [](std::string_view part) {
    return !part.empty() && part.find_first_not_of(". ") == std::string_view::npos;
  });
}

// Checks what can be known about an external data location without the file
// system: it must be a non-empty relative path that names a file and has no '..'
// component. Returns the location in normalized form. Used both when reading
// and when writing external data.
std::filesystem::path check_external_data_location_syntax(
    const std::string& base_dir,
    const std::string& location,
    const std::string& tensor_name) {
  auto file_path = utf8_to_path(location);
  if (file_path.empty()) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "Location of external TensorProto ( tensor name: ",
        tensor_name,
        ") should not be empty.");
  }
  if (file_path.is_absolute()) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "Location of external TensorProto ( tensor name: ",
        tensor_name,
        ") should be a relative path, but it is an absolute path: ",
        location);
  }
  auto relative_path = file_path.lexically_normal().make_preferred();
  if (has_parent_directory_component(relative_path)) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "Data of TensorProto ( tensor name: ",
        tensor_name,
        ") should be file inside '",
        base_dir,
        "', but '",
        location,
        "' points outside the directory.");
  }
  // ".", "sub/..", "weights/" and "weights\\" name a directory, never a file.
  // On POSIX, has_filename() does not recognize '\\' as a separator.
  if (relative_path == std::filesystem::path(".") || !relative_path.has_filename() || location.back() == '\\') {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "Location of external TensorProto ( tensor name: ",
        tensor_name,
        ") should name a file, but it is: ",
        location);
  }
  if (has_dots_and_spaces_component(relative_path)) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "Location of external TensorProto ( tensor name: ",
        tensor_name,
        ") should name a file, but it has a component made only of dots and spaces, which Windows reads as the "
        "current or parent directory: ",
        location);
  }
  return relative_path;
}

// Without the file system, the containment check of resolve_external_data_location
// cannot run. What can be decided from the path alone: on Windows, a location with
// a root directory ("\data.bin") or with a drive other than the model directory's
// ("D:data.bin") resolves outside the model directory. A location on the model
// directory's drive ("C:data.bin" next to "C:\models\") resolves inside it.
void check_external_data_location_in_model_dir(
    const std::string& base_dir,
    const std::filesystem::path& relative_path,
    const std::string& tensor_name) {
  if (relative_path.has_root_directory() ||
      (relative_path.has_root_name() && relative_path.root_name() != utf8_to_path(base_dir).root_name())) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "Tensor ",
        tensor_name,
        " external data resolves outside model directory.");
  }
}

static std::vector<std::filesystem::path> external_data_path_components(
    const std::string& base_dir,
    const std::string& location,
    const std::string& tensor_name) {
  auto relative_path = check_external_data_location_syntax(base_dir, location, tensor_name);
  check_external_data_location_in_model_dir(base_dir, relative_path, tensor_name);
  relative_path = relative_path.relative_path();
  std::vector<std::filesystem::path> components;
  for (const auto& component : relative_path) {
    if (component.empty() || component == std::filesystem::path(".") || component == std::filesystem::path("..")) {
      fail_check_with_code(
          error_code::kExternalDataLocationInvalid,
          "External data location for tensor ",
          tensor_name,
          " has an invalid path component: ",
          location);
    }
    components.push_back(component);
  }
  if (components.empty()) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "External data location for tensor ",
        tensor_name,
        " does not name a file: ",
        location);
  }
  return components;
}

struct ExternalDataContext::Impl {
  std::string base_dir;
#ifdef _WIN32
  HANDLE base_handle = INVALID_HANDLE_VALUE;
#else
  int base_fd = -1;
#endif
};

#ifdef _WIN32

using NtCreateFileFn = NTSTATUS(NTAPI*)(
    PHANDLE,
    ACCESS_MASK,
    POBJECT_ATTRIBUTES,
    PIO_STATUS_BLOCK,
    PLARGE_INTEGER,
    ULONG,
    ULONG,
    ULONG,
    ULONG,
    PVOID,
    ULONG);
using RtlNtStatusToDosErrorFn = ULONG(NTAPI*)(NTSTATUS);

static NtCreateFileFn get_nt_create_file() {
  static const auto nt_create_file =
      reinterpret_cast<NtCreateFileFn>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtCreateFile"));
  if (nt_create_file == nullptr) {
    fail_check("Secure external-data opening is unavailable on this Windows system.");
  }
  return nt_create_file;
}

static ULONG nt_status_to_dos_error(NTSTATUS status) {
  static const auto convert = reinterpret_cast<RtlNtStatusToDosErrorFn>(
      GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlNtStatusToDosError"));
  return convert == nullptr ? ERROR_GEN_FAILURE : convert(status);
}

static HANDLE open_windows_component(
    HANDLE root,
    const std::filesystem::path& component,
    ACCESS_MASK access,
    ULONG disposition,
    ULONG options,
    ULONG share_access,
    const std::string& tensor_name) {
  const auto& name = component.native();
  if (name.size() > std::numeric_limits<USHORT>::max() / sizeof(wchar_t)) {
    fail_check("External data path component is too long for tensor ", tensor_name, ".");
  }
  UNICODE_STRING unicode_name{};
  unicode_name.Buffer = const_cast<PWSTR>(name.data());
  unicode_name.Length = static_cast<USHORT>(name.size() * sizeof(wchar_t));
  unicode_name.MaximumLength = unicode_name.Length;
  OBJECT_ATTRIBUTES attributes{};
  attributes.Length = sizeof(attributes);
  attributes.RootDirectory = root;
  attributes.ObjectName = &unicode_name;
  attributes.Attributes = OBJ_CASE_INSENSITIVE;
  IO_STATUS_BLOCK io_status{};
  HANDLE handle = INVALID_HANDLE_VALUE;
  const NTSTATUS status = get_nt_create_file()(
      &handle,
      access,
      &attributes,
      &io_status,
      nullptr,
      FILE_ATTRIBUTE_NORMAL,
      share_access,
      disposition,
      options,
      nullptr,
      0);
  if (status < 0 || handle == INVALID_HANDLE_VALUE) {
    const ULONG error = nt_status_to_dos_error(status);
    fail_check_with_code(
        error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND ? error_code::kExternalDataFileMissing
                                                                       : error_code::kExternalDataLocationInvalid,
        "Cannot open external data path component for tensor ",
        tensor_name,
        ".");
  }
  return handle;
}

static BY_HANDLE_FILE_INFORMATION
validate_windows_handle(HANDLE handle, const std::string& tensor_name, bool expect_directory) {
  BY_HANDLE_FILE_INFORMATION info{};
  if (!GetFileInformationByHandle(handle, &info)) {
    fail_check("Tensor ", tensor_name, " external data: cannot query file information.");
  }
  if (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "Tensor ",
        tensor_name,
        " external data path contains a reparse point.");
  }
  const bool is_directory = (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
  if (is_directory != expect_directory) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid,
        "Tensor ",
        tensor_name,
        " external data path has an unexpected file type.");
  }
  return info;
}

static HANDLE open_windows_leaf(
    HANDLE base_handle,
    const std::vector<std::filesystem::path>& components,
    ACCESS_MASK access,
    ULONG disposition,
    const std::string& tensor_name,
    bool open_reparse_point = true) {
  std::vector<HANDLE> directories;
  ONNX_NAMESPACE::ScopeExit close_directories([&]() noexcept {
    for (HANDLE handle : directories) {
      CloseHandle(handle);
    }
  });
  HANDLE parent = base_handle;
  for (size_t i = 0; i + 1 < components.size(); ++i) {
    HANDLE directory = open_windows_component(
        parent,
        components[i],
        FILE_TRAVERSE | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
        FILE_OPEN,
        FILE_DIRECTORY_FILE | FILE_OPEN_REPARSE_POINT | FILE_SYNCHRONOUS_IO_NONALERT,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        tensor_name);
    ScopedHandle directory_guard(directory);
    validate_windows_handle(directory, tensor_name, true);
    directories.push_back(directory);
    directory_guard.release();
    parent = directory;
  }
  return open_windows_component(
      parent,
      components.back(),
      access,
      disposition,
      FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT | (open_reparse_point ? FILE_OPEN_REPARSE_POINT : 0),
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      tensor_name);
}

#else // POSIX

static int directory_open_flags() {
  int flags = O_DIRECTORY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_PATH
  flags |= O_PATH;
#elif defined(O_SEARCH)
  flags |= O_SEARCH;
#else
  flags |= O_RDONLY;
#endif
  return flags;
}

static int open_posix_parent(
    int base_fd,
    const std::vector<std::filesystem::path>& components,
    const std::string& tensor_name,
    std::vector<int>& opened_directories) {
  int parent = base_fd;
  for (size_t i = 0; i + 1 < components.size(); ++i) {
    int flags = directory_open_flags();
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#else
    fail_check("Secure external-data opening requires O_NOFOLLOW for tensor ", tensor_name, ".");
#endif
    const int directory = openat(parent, components[i].c_str(), flags);
    if (directory < 0) {
      fail_check_with_code(
          errno == ENOENT ? error_code::kExternalDataFileMissing : error_code::kExternalDataLocationInvalid,
          "Cannot open external data directory component for tensor ",
          tensor_name,
          ".");
    }
    ScopedFd directory_guard(directory);
    opened_directories.push_back(directory);
    directory_guard.release();
    parent = directory;
  }
  return parent;
}

static void validate_posix_file(const struct stat& file_stat, const std::string& tensor_name) {
  if (!S_ISREG(file_stat.st_mode)) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid, "Tensor ", tensor_name, " external data is not a regular file.");
  }
  if (file_stat.st_nlink > 1) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid, "Tensor ", tensor_name, " external data has multiple hard links.");
  }
}

#endif

ExternalDataContext::ExternalDataContext(const std::string& base_dir) : impl_(std::make_unique<Impl>()) {
  const auto normalized_base_dir = base_dir.empty() ? std::string(".") : base_dir;
  std::error_code ec;
  impl_->base_dir = path_to_utf8(std::filesystem::absolute(utf8_to_path(normalized_base_dir), ec));
  if (ec) {
    fail_check("Cannot resolve external data base directory: ", ec.message());
  }
#ifdef _WIN32
  HANDLE base_handle = CreateFileW(
      utf8_to_path(impl_->base_dir).native().c_str(),
      FILE_TRAVERSE | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      nullptr,
      OPEN_EXISTING,
      FILE_FLAG_BACKUP_SEMANTICS,
      nullptr);
  if (base_handle == INVALID_HANDLE_VALUE) {
    fail_check("Cannot open external data base directory.");
  }
  ScopedHandle base_guard(base_handle);
  validate_windows_handle(base_handle, "base directory", true);
  impl_->base_handle = base_guard.release();
#else
  impl_->base_fd = ::open(impl_->base_dir.c_str(), directory_open_flags());
  if (impl_->base_fd < 0) {
    fail_check("Cannot open external data base directory.");
  }
  struct stat base_stat{};
  if (fstat(impl_->base_fd, &base_stat) != 0 || !S_ISDIR(base_stat.st_mode)) {
    close(impl_->base_fd);
    impl_->base_fd = -1;
    fail_check("External data base path is not a directory.");
  }
#endif
}

ExternalDataContext::~ExternalDataContext() {
#ifdef _WIN32
  if (impl_->base_handle != INVALID_HANDLE_VALUE) {
    CloseHandle(impl_->base_handle);
  }
#else
  if (impl_->base_fd >= 0) {
    close(impl_->base_fd);
  }
#endif
}

int64_t ExternalDataContext::open(const std::string& location, const std::string& tensor_name, bool read_only) const {
  const auto components = external_data_path_components(impl_->base_dir, location, tensor_name);
#ifdef _WIN32
  HANDLE handle = open_windows_leaf(
      impl_->base_handle,
      components,
      (read_only ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE)) | SYNCHRONIZE,
      read_only ? FILE_OPEN : FILE_OPEN_IF,
      tensor_name);
  ScopedHandle guard(handle);
  const auto info = validate_windows_handle(handle, tensor_name, false);
  if (info.nNumberOfLinks > 1) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid, "Tensor ", tensor_name, " external data has multiple hard links.");
  }
  HANDLE released = guard.release();
  const int flags = read_only ? (_O_RDONLY | _O_BINARY) : (_O_RDWR | _O_BINARY);
  const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(released), flags);
  if (fd < 0) {
    CloseHandle(released);
    fail_check("Cannot convert external data handle to fd for tensor ", tensor_name, ".");
  }
  return static_cast<int64_t>(fd);
#else
  std::vector<int> directories;
  ONNX_NAMESPACE::ScopeExit close_directories([&]() noexcept {
    for (int fd : directories) {
      close(fd);
    }
  });
  const int parent = open_posix_parent(impl_->base_fd, components, tensor_name, directories);
  // A FIFO must reach regular-file validation without waiting for a writer.
  int flags = read_only ? (O_RDONLY | O_NONBLOCK) : (O_CREAT | O_RDWR);
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#else
  fail_check("Secure external-data opening requires O_NOFOLLOW for tensor ", tensor_name, ".");
#endif
  const int fd = read_only ? openat(parent, components.back().c_str(), flags)
                           : openat(parent, components.back().c_str(), flags, 0600);
  if (fd < 0) {
    fail_check_with_code(
        read_only && errno == ENOENT ? error_code::kExternalDataFileMissing : error_code::kExternalDataLocationInvalid,
        "Cannot open external data for tensor ",
        tensor_name,
        ".");
  }
  ScopedFd guard(fd);
  struct stat fd_stat{};
  if (fstat(fd, &fd_stat) != 0) {
    fail_check("Tensor ", tensor_name, " external data: fstat failed.");
  }
  validate_posix_file(fd_stat, tensor_name);
  return guard.release();
#endif
}

std::uintmax_t ExternalDataContext::file_size(const std::string& location, const std::string& tensor_name) const {
  const auto components = external_data_path_components(impl_->base_dir, location, tensor_name);
#ifdef _WIN32
  ScopedHandle handle(
      open_windows_leaf(impl_->base_handle, components, FILE_READ_ATTRIBUTES | SYNCHRONIZE, FILE_OPEN, tensor_name));
  const auto info = validate_windows_handle(handle.get(), tensor_name, false);
  if (info.nNumberOfLinks > 1) {
    fail_check_with_code(
        error_code::kExternalDataLocationInvalid, "Tensor ", tensor_name, " external data has multiple hard links.");
  }
  ULARGE_INTEGER size{};
  size.HighPart = info.nFileSizeHigh;
  size.LowPart = info.nFileSizeLow;
  return static_cast<std::uintmax_t>(size.QuadPart);
#else
  std::vector<int> directories;
  ONNX_NAMESPACE::ScopeExit close_directories([&]() noexcept {
    for (int fd : directories) {
      close(fd);
    }
  });
  const int parent = open_posix_parent(impl_->base_fd, components, tensor_name, directories);
  struct stat file_stat{};
  if (fstatat(parent, components.back().c_str(), &file_stat, AT_SYMLINK_NOFOLLOW) != 0) {
    fail_check_with_code(
        errno == ENOENT ? error_code::kExternalDataFileMissing : error_code::kExternalDataLocationInvalid,
        "Cannot inspect external data for tensor ",
        tensor_name,
        ".");
  }
  validate_posix_file(file_stat, tensor_name);
  if (file_stat.st_size < 0) {
    fail_check("Tensor ", tensor_name, " external data has a negative file size.");
  }
  return static_cast<std::uintmax_t>(file_stat.st_size);
#endif
}

// NOLINTEND(bugprone-signed-bitwise)

} // namespace ONNX_NAMESPACE::checker
