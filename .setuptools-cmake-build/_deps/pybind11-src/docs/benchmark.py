from __future__ import annotations

import datetime as dt
import os
import random

nfns = 4  # Functions per class
nargs = 4  # Arguments per function


def generate_dummy_code_pybind11(nclasses=10):
    decl = ""
    bindings = ""

    for cl in range(nclasses):
        decl += f"class cl{cl:03};\n"
    decl += "\n"

    for cl in range(nclasses):
        decl += f"class {cl:03} {{\n"
        decl += "public:\n"
        bindings += f'    py::class_<cl{cl:03}>(m, "cl{cl:03}")\n'
        for fn in range(nfns):
            ret = random.randint(0, nclasses - 1)
            params = [random.randint(0, nclasses - 1) for i in range(nargs)]
            decl += f"    cl{ret:03} *fn_{fn:03}("
            decl += ", ".join(f"cl{p:03} *" for p in params)
            decl += ");\n"
            bindings += f'        .def("fn_{fn:03}", &cl{cl:03}::fn_{fn:03})\n'
        decl += "};\n\n"
        bindings += "        ;\n"

    result = "#include <pybind11/pybind11.h>\n\n"
    result += "namespace py = pybind11;\n\n"
    result += decl + "\n"
    result += "PYBIND11_MODULE(example, m) {\n"
    result += bindings
    result += "}"
    return result


def generate_dummy_code_boost(nclasses=10):
    decl = ""
    bindings = ""

    for cl in range(nclasses):
        decl += f"class cl{cl:03};\n"
    decl += "\n"

    for cl in range(nclasses):
        decl += "class cl%03i {\n" % cl
        decl += "public:\n"
        bindings += f'    py::class_<cl{cl:03}>("cl{cl:03}")\n'
        for fn in range(nfns):
            ret = random.randint(0, nclasses - 1)
            params = [random.randint(0, nclasses - 1) for i in range(nargs)]
            decl += f"    cl{ret:03} *fn_{fn:03}("
            decl += ", ".join(f"cl{p:03} *" for p in params)
            decl += ");\n"
            bindings += f'        .def("fn_{fn:03}", &cl{cl:03}::fn_{fn:03}, py::return_value_policy<py::manage_new_object>())\n'
        decl += "};\n\n"
        bindings += "        ;\n"

    result = "#include <boost/python.hpp>\n\n"
    result += "namespace py = boost::python;\n\n"
    result += decl + "\n"
    result += "BOOST_PYTHON_MODULE(example) {\n"
    result += bindings
    result += "}"
    return result


for codegen in [generate_dummy_code_pybind11, generate_dummy_code_boost]:
    print("{")
    for i in range(10):
        nclasses = 2**i
        with open("test.cpp", "w") as f:
            f.write(codegen(nclasses))
        n1 = dt.datetime.now()
        os.system(
            "g++ -Os -shared -rdynamic -undefined dynamic_lookup "
            "-fvisibility=hidden -std=c++14 test.cpp -I include "
            "-I /System/Library/Frameworks/Python.framework/Headers -o test.so"
        )
        n2 = dt.datetime.now()
        elapsed = (n2 - n1).total_seconds()
        size = os.stat("test.so").st_size
        print("   {%i, %f, %i}," % (nclasses * nfns, elapsed, size))
    print("}")
