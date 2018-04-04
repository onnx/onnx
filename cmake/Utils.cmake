###
# Removes common indentation from a block of text to produce code suitable for
# setting to `python -c`, or using with pycmd. This allows multiline code to be
# nested nicely in the surrounding code structure.
#
# This function respsects PYTHON_EXECUTABLE if it defined, otherwise it uses
# `python` and hopes for the best. An error will be thrown if it is not found.
#
# Args:
#     outvar : variable that will hold the stdout of the python command
#     text   : text to remove indentation from
#
function(dedent outvar text)
  # Use PYTHON_EXECUTABLE if it is defined, otherwise default to python
  if ("${PYTHON_EXECUTABLE}" STREQUAL "")
    set(_python_exe "python")
  else()
    set(_python_exe "${PYTHON_EXECUTABLE}")
  endif()
  set(_fixup_cmd "import sys; from textwrap import dedent; print(dedent(sys.stdin.read()))")
  # Use echo to pipe the text to python's stdinput. This prevents us from
  # needing to worry about any sort of special escaping.
  execute_process(
    COMMAND echo "${text}"
    COMMAND "${_python_exe}" -c "${_fixup_cmd}"
    RESULT_VARIABLE _dedent_exitcode
    OUTPUT_VARIABLE _dedent_text)
  if(NOT ${_dedent_exitcode} EQUAL 0)
    message(ERROR " Failed to remove indentation from: \n\"\"\"\n${text}\n\"\"\"
    Python dedent failed with error code: ${_dedent_exitcode}")
    message(FATAL_ERROR " Python dedent failed with error code: ${_dedent_exitcode}")
  endif()
  # Remove supurflous newlines (artifacts of print)
  string(STRIP "${_dedent_text}" _dedent_text)
  set(${outvar} "${_dedent_text}" PARENT_SCOPE)
endfunction()

###
# Helper function to run `python -c "<cmd>"` and capture the results of stdout
#
# Runs a python command and populates an outvar with the result of stdout.
# Common indentation in the text of `cmd` is removed before the command is
# executed, so the caller does not need to worry about indentation issues.
#
# This function respsects PYTHON_EXECUTABLE if it defined, otherwise it uses
# `python` and hopes for the best. An error will be thrown if it is not found.
#
# Args:
#     outvar : variable that will hold the stdout of the python command
#     cmd    : text representing a (possibly multiline) block of python code
#
function(pycmd outvar cmd)
  dedent(_dedent_cmd "${cmd}")
  # Use PYTHON_EXECUTABLE if it is defined, otherwise default to python
  if ("${PYTHON_EXECUTABLE}" STREQUAL "")
    set(_python_exe "python")
  else()
    set(_python_exe "${PYTHON_EXECUTABLE}")
  endif()
  # run the actual command
  execute_process(
    COMMAND "${_python_exe}" -c "${_dedent_cmd}"
    RESULT_VARIABLE _exitcode
    OUTPUT_VARIABLE _output)
  if(NOT ${_exitcode} EQUAL 0)
    message(ERROR " Failed when running python code: \"\"\"\n${_dedent_cmd}\n\"\"\"")
    message(FATAL_ERROR " Python command failed with error code: ${_exitcode}")
  endif()
  # Remove supurflous newlines (artifacts of print)
  string(STRIP "${_output}" _output)
  set(${outvar} "${_output}" PARENT_SCOPE)
endfunction()
