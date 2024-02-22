# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Automates the generation of ONNX operators."""
import difflib
import importlib
import inspect
import keyword
import os
import pathlib
import re
import shutil
import sys
import textwrap
from typing import Any

import jinja2
import numpy as np
from sphinx.util import logging

import onnx
from onnx.backend.test.case.base import _Exporter
from onnx.defs import OpSchema

REPO_DOCS_EXCLUDE = {
    "Changelog-ml.md",
    "Changelog.md",
    "CIPipelines.md",
    "CONTRIBUTING.md",
    "Operators-ml.md",
    "Operators.md",
    "Relicensing.md",
    "TestCoverage-ml.md",
    "TestCoverage.md",
}


def _get_diff_template():
    return jinja2.Template(
        textwrap.dedent(
            """
            <div id="{{ div_name }}"></div>
            <link rel="stylesheet" type="text/css" href="../_static/diff2html.min.css" />
            <script type="text/javascript" src="../_static/diff2html-ui-slim.min.js"></script>
            <script>
            const diffString = `
            --- a/{{ op_name }}{{ version1 }}
            +++ b/{{ op_name }}{{ version2 }}
            @@ -1 +1 @@
            {{ diff_content }}
            `;

            document.addEventListener('DOMContentLoaded', function () {
            var targetElement = document.getElementById('{{ div_name }}');
            var configuration = {
                drawFileList: true,
                fileListToggle: false,
                fileListStartVisible: false,
                fileContentToggle: false,
                matching: 'lines',
                outputFormat: 'line-by-line',
                synchronisedScroll: true,
                highlight: true,
                renderNothingWhenEmpty: false,
            };
            var diff2htmlUi = new Diff2HtmlUI(targetElement, diffString, configuration);
            diff2htmlUi.draw();
            diff2htmlUi.highlightCode();
            });
            </script>
            """
        ),
        autoescape=True,
    )


def _get_ops_template():
    return jinja2.Template(
        """\
{% for sch in schemas %}

.. tag-diff-insert.

(l-onnx-op{{sch.domain.lower().replace(".", "-")}}-{{sch.name.lower()}}-{{str(sch.since_version)}})=

## {{format_name_with_domain(sch)}}

### Version

- **name**: [{{sch.name}} (GitHub)]({{build_doc_url(sch)}}{{sch.name}})
- **domain**: `{% if sch.domain == '' %}main{% else %}{{sch.domain}}{% endif %}`
- **since_version**: `{{sch.since_version}}`
- **function**: `{{sch.has_function or sch.has_context_dependent_function}}`
- **support_level**: `{{sch.support_level}}`
- **shape inference**: `{{sch.has_type_and_shape_inference_function}}`

{% if sch.support_level == OpSchema.SupportType.EXPERIMENTAL %}
No versioning maintained for experimental ops.
{% else %}
This version of the operator has been {% if
sch.deprecated %}deprecated{% else %}available{% endif %}
**since version {{sch.since_version}}{% if
sch.domain %} of domain {{sch.domain}}{% endif %}**.
{% if len(sch.versions) > 1 %}
Other versions of this operator:
{% for v in sch.version[:-1] %} {{v}} {% endfor %}
{% endif %}
{% endif %}

### Summary

{{process_documentation(sch.doc)}}

{% if sch.has_function %}

#### Function Body

The function definition for this operator.

```
{{get_function_body(sch)}}
```
{% endif %}
{% if sch.attributes %}

### Attributes

{% for _, attr in sorted(sch.attributes.items())
%}* **{{attr.name}} - {{str(attr.type).split('.')[-1]}}**{%
  if attr.required %} (required){% endif %} {%
  if attr.default_value %}{{clean_default_value(attr)}}{%
  endif %}:

{{text_indent(attr.description, 2)}}

{% endfor %}
{% endif %}
{% if sch.inputs %}

### Inputs

{% if sch.min_input != sch.max_input %}Between {{sch.min_input
}} and {{sch.max_input}} inputs.
{% endif %}
{% for ii, inp in enumerate(sch.inputs) %}
- **{{getname(inp, ii)}}**{{format_option(inp)}} - **{{inp.type_str}}**:

{{text_indent(inp.description, 2)}}{% endfor %}
{% endif %}
{% if sch.outputs %}

### Outputs

{% if sch.min_output != sch.max_output %}Between {{sch.min_output
}} and {{sch.max_output}} outputs.
{% endif %}
{% for ii, out in enumerate(sch.outputs) %}
- **{{getname(out, ii)}}**{{format_option(out)}} - **{{out.type_str}}**:

{{text_indent(out.description, 2)}}{% endfor %}
{% endif %}
{% if sch.type_constraints %}

### Type Constraints

{% for ii, type_constraint in enumerate(sch.type_constraints)
%}* {{get_constraint(type_constraint, ii)}}:

{{text_indent(type_constraint.description, 2)}}
{% endfor %}
{% endif %}
{% if examples and is_last_schema(sch): %}

### Examples

{% for example, code in examples.items(): %}

#### {{ example }}

```python
{{ format_example(code) }}
```
{% endfor %}
{% endif %}
{% endfor %}""",
    autoescape=False,
)


def _get_main_template():
    return jinja2.Template(
        textwrap.dedent(
            """
            .. _l-onnx-operators:

            {{ title }}
            {{ "=" * len(title) }}

            Lists out all the ONNX operators. For each operator, lists out the usage guide,
            parameters, examples, and line-by-line version history.
            This section also includes tables detailing each operator
            with its versions, as done in `Operators.md
            <https://github.com/onnx/onnx/blob/main/docs/Operators.md>`_.

            All examples end by calling function `expect`.
            which checks a runtime produces the expected output for this example.
            One implementation based on `onnxruntime <https://onnxruntime.ai/>`_
            can be found at :ref:`l-function-expect`.

            .. toctree::
                :hidden:

                ../expect_onnxruntime
                {% for p in pages %}{{ os.path.split(p)[-1] }}
                {% endfor %}

            .. tabs::

                {% for t in tabs %}.. tab:: {{ t.domain_name }}
                    {{ t.render(indent="        ") }}
                {% endfor %}
            """
        ),
        autoescape=True,
    )


def _clean_unicode(text):
    text = text.replace("&#34;", '"')
    text = text.replace("&#8212;", "-")
    text = text.replace("&#160;", " ")
    text = text.replace("&#39;", "'")
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    return text


_template_diff = _get_diff_template()
_template_operator = _get_ops_template()
_template_main = _get_main_template()
_all_schemas_with_history = None


_attribute_conversion_functions = {
    onnx.AttributeProto.FLOAT: lambda att: np.float32(att.f),
    onnx.AttributeProto.FLOATS: lambda att: [np.float32(f) for f in att.floats],
    # AttributeProto.GRAPH(5)
    # AttributeProto.GRAPHS(10)
    onnx.AttributeProto.INT: lambda att: int(att.i),
    onnx.AttributeProto.INTS: lambda att: [int(i) for i in att.ints],
    # AttributeProto.SPARSE_TENSOR(11)
    # AttributeProto.SPARSE_TENSORS(12)
    onnx.AttributeProto.STRING: lambda att: att.s.decode("utf-8"),
    onnx.AttributeProto.STRINGS: lambda att: [s.decode("utf-8") for s in att.strings],
    onnx.AttributeProto.TENSOR: lambda att: onnx.numpy_helper.to_array(att.t),
    # AttributeProto.TENSORS(9)
    # onnx.AttributeProto.TYPE_PROTO: lambda att: OnnxType(att.tp),
    # AttributeProto.TYPE_PROTOS(14)
}


def _populate_all_schemas_with_history():
    res: dict[str, Any] = {}
    for schema in onnx.defs.get_all_schemas_with_history():
        domain = schema.domain
        version = schema.since_version
        name = schema.name
        if domain not in res:
            res[domain] = {}
        if name not in res[domain]:
            res[domain][name] = {}
        res[domain][name][version] = schema

    return res


def _get_all_schemas_with_history():
    global _all_schemas_with_history
    if _all_schemas_with_history is None:
        _all_schemas_with_history = _populate_all_schemas_with_history()
    return _all_schemas_with_history


def get_operator_schemas(op_name, version=None, domain=None):
    """
    Returns all schemas mapped to an operator name.
    :param op_name: name of the operator
    :param version: version
    :param domain: domain
    :return: list of schemas
    """
    if version == "last" and op_name is not None:
        if domain is not None:
            return [onnx.defs.get_schema(op_name, domain=domain)]
    all_schemas = _get_all_schemas_with_history()
    if domain is None:
        domains = []
        for dom, ops in all_schemas.items():
            if op_name is None or op_name in ops:
                domains.append(dom)
    else:
        domains = [domain]

    # schemas
    sch = []
    for dom in domains:
        ops = all_schemas[dom]
        if op_name is None:
            for op, v in ops.items():
                if version is None:
                    sch.extend(v.values())
                elif version == "last" and (dom == "" or "onnx" in dom):
                    try:
                        sch.append(onnx.defs.get_schema(op, domain=dom))
                    except onnx.defs.SchemaError:
                        sch.append(v[max(v)])
                elif version == "last":
                    sch.append(v[max(v)])
                else:
                    sch.append(v[version])
        elif op_name in ops:
            if version is None:
                sch.extend(ops[op_name].values())
            elif version in ops[op_name]:
                sch.append(ops[op_name][version])

    # sort
    vals = [(s.domain, s.name, -s.since_version, s) for s in sch]
    vals.sort()
    return [v[-1] for v in vals]


def get_markdown_doc(
    folder,
    op_name=None,
    domain=None,
    version="last",
    clean=True,
    diff=False,
    example=False,
):
    """
    Returns a documentation in Markdown format
    for all :class:`OnnxOperator`.

    :param op_name: operator name of None for all
    :param domain: domain
    :param version: version, None for all, `'last'` for the most recent one
    :param clean: clean empty lines
    :param diff: highlights differences between two versions
    :param example: add example to the documentation
    :return: string
    """
    schemas = get_operator_schemas(op_name, domain=domain, version=version)

    def format_name_with_domain(sch):
        if version == "last":
            if sch.domain:
                return f"{sch.name} ({sch.domain})"
            return sch.name
        if sch.domain:
            return f"{sch.name} - {sch.since_version} ({sch.domain})"
        return f"{sch.name} - {sch.since_version}"

    def format_option(obj):
        opts = []
        if OpSchema.FormalParameterOption.Optional == obj.option:
            opts.append("optional")
        elif OpSchema.FormalParameterOption.Variadic == obj.option:
            opts.append("variadic")
        if getattr(obj, "is_homogeneous", False):
            opts.append("heterogeneous")
        if opts:
            return f" ({', '.join(opts)})"
        return ""

    def format_example(code):
        return code

    def get_constraint(const, ii):
        if const.type_param_str:
            name = const.type_param_str
        else:
            name = str(ii)
        name = f"**{name}** in ("
        if const.allowed_type_strs:
            types = [f"`{type_str}`" for type_str in sorted(const.allowed_type_strs)]
            text = ", ".join(types)
            name += " " + text + " )"
        return name

    def getname(obj, i):
        name = obj.name
        if len(name) == 0:
            return str(i)
        return name

    def process_documentation(doc):
        if doc is None:
            doc = ""
        if not isinstance(doc, str):
            raise TypeError(f"doc must be a string not {type(doc)!r} - {doc + 42!r}.")
        main_docs_url = "https://github.com/onnx/onnx/blob/main/"
        rep = {
            "[the doc](IR.md)": f"[ONNX IR]({main_docs_url}docs/IR.md)",
            "[the doc](Broadcasting.md)": f"[Broadcasting in ONNX]({main_docs_url}docs/Broadcasting.md)",
        }
        for key, value in rep.items():
            doc = doc.replace(key, value)
        return textwrap.dedent(doc)

    def build_doc_url(sch):
        doc_url = "https://github.com/onnx/onnx/blob/main/docs/Operators"
        if "ml" in sch.domain:
            doc_url += "-ml"
        doc_url += ".md"
        doc_url += "#"
        if sch.domain not in (None, "", "ai.onnx"):
            doc_url += sch.domain + "."
        return doc_url

    def format_default_value(value):
        if isinstance(value, float):
            formatted = str(np.round(value, 5))
            # use default formatting, unless too long.
            if len(formatted) > 10:
                formatted = f"({value:e})"
            return formatted
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        return str(value)

    def clean_default_value(attr):
        if not attr.default_value.name:
            return ""
        default_value = onnx.helper.get_attribute_value(attr.default_value)
        if isinstance(default_value, onnx.AttributeProto) and hasattr(
            default_value, "default_value"
        ):
            if attr.type in _attribute_conversion_functions:
                sval = _attribute_conversion_functions[attr.type](default_value)
                return f"(default is `{sval!r}`)"

        if isinstance(default_value, list):
            sval = [format_default_value(val) for val in default_value]
        else:
            sval = format_default_value(default_value)
        return f"(default is `{sval!r}`)"

    def text_indent(text: str, indent: int) -> str:
        s = " " * indent
        return textwrap.indent(text, s)

    def get_function_body(schema: OpSchema) -> str:
        return onnx.printer.to_text(schema.function_body)

    examples = get_onnx_example(op_name, domain) if example else {}
    docs = _template_operator.render(
        schemas=schemas,
        OpSchema=OpSchema,
        len=len,
        getattr=getattr,
        sorted=sorted,
        format_option=format_option,
        get_constraint=get_constraint,
        getname=getname,
        enumerate=enumerate,
        format_name_with_domain=format_name_with_domain,
        process_documentation=process_documentation,
        build_doc_url=build_doc_url,
        text_indent=text_indent,
        str=str,
        clean_default_value=clean_default_value,
        examples=examples,
        format_example=format_example,
        is_last_schema=is_last_schema,
        get_function_body=get_function_body,
    )

    d_links = {}
    for schema in schemas:
        sdom = schema.domain.replace(".", "-")
        d_links[
            schema.since_version
        ] = f"l-onnx-op{sdom}-{schema.name.lower()}-{schema.since_version}"

    if diff:
        lines = docs.split("\n")
        new_lines = [""]
        for line in lines:
            line = line.rstrip("\r\t ")
            if len(line) == 0 and len(new_lines[-1]) == 0:
                continue
            new_lines.append(line)
        docs = "\n".join(new_lines)
        docs, d_links_diff = _insert_diff(
            folder,
            docs,
            ".. tag-diff-insert.",
            op_name=op_name,
            version=version,
            domain=domain,
        )
        d_links.update(d_links_diff)

    if clean:
        lines = docs.split("\n")
        new_lines = [""]
        for line in lines:
            line = line.rstrip("\r\t ")
            if len(line) == 0 and len(new_lines[-1]) == 0:
                continue
            new_lines.append(line)
        docs = "\n".join(new_lines)

    return docs, d_links, len(examples)


def _insert_diff(
    folder, docs, split=".. tag-diff-insert.", op_name=None, version=None, domain=None
):
    """
    Splits a using `split`, insert HTML differences between pieces.
    The function relies on package `pyquickhelper`.
    """
    doc_parts = docs.split(split)
    if len(doc_parts) <= 1:
        return docs

    reg = re.compile("([A-Z][A-Za-z0-9_]*) - ([0-9]+)")

    d_links = {}
    pieces = [doc_parts[0]]
    mds = []
    for i in range(1, len(doc_parts)):
        spl1 = doc_parts[i - 1].strip("\n ")
        spl2 = doc_parts[i].strip("\n ")
        vers1 = reg.findall(spl1)
        vers2 = reg.findall(spl2)

        spl1 = spl1.split("### Examples")[0].replace("`", "")
        spl2 = spl2.split("### Examples")[0].replace("`", "")
        spl1 = spl1.split("### Summary")[-1].strip("\n ")
        spl2 = spl2.split("### Summary")[-1].strip("\n ")
        if len(spl1) < 5 or len(spl2) < 5:
            pieces.append(doc_parts[i])
            continue
        if not vers1:
            raise ValueError(f"Unable to find version {version!r} in\n{spl1}")
        if not vers2:
            raise ValueError(f"Unable to find version {version!r} in\n{spl2}")
        v2 = vers2[0][1]
        v1 = vers1[0][1]

        if not mds:
            mds.append(
                (v1, textwrap.dedent(spl1.strip(" \n\r\t")).splitlines(keepends=True))
            )
        mds.append(
            (v2, textwrap.dedent(spl2.strip(" \n\r\t")).splitlines(keepends=True))
        )

        if len(mds) > 1:
            show_diff_toc = True
        else:
            show_diff_toc = False

        if show_diff_toc:
            pieces.append("```{toctree}")

        for di in range(len(mds) - 1):
            dj = len(mds) - 1

            v1, s1 = mds[di]
            v2, s2 = mds[dj]
            differ = difflib.Differ()
            result = list(differ.compare(s2, s1))
            raw = "".join(result)

            diff = _template_diff.render(
                op_name=op_name,
                version1=v2,
                version2=v1,
                div_name=f"div_{op_name}_{i}",
                diff_content=raw,
            )
            diff = _clean_unicode(diff)

            title = f"{op_name} - {v2} vs {v1}"

            name = f"text_diff_{op_name}_{v2}_{v1}"
            domain_str = domain.replace(".", "-")
            link = f"l-onnx-op{domain_str}-{op_name.lower()}-d{v2}-{v1}"
            d_links[int(v2), int(v1)] = link
            content = "\n".join(
                [
                    "",
                    f".. _{link}:",
                    "",
                    title,
                    "=" * len(title),
                    "",
                    "Next section compares an older to a newer version of the same operator ",
                    "after both definition are converted into markdown text.",
                    "Green means an addition to the newer version, red means a deletion.",
                    "Anything else is unchanged.",
                    "",
                    ".. raw:: html",
                    "",
                    textwrap.indent(diff, "    "),
                ]
            )
            filename = os.path.join(folder, name + ".rst")
            pathlib.Path(filename).write_text(content, encoding="utf-8")
            # Add diff page to the toctree using myst syntax
            pieces.append(name)

        if show_diff_toc:
            # End the toctree
            pieces.append("```")

        pieces.extend(["", doc_parts[i]])

    return "\n".join(pieces), d_links


def pascal_to_snake_case(name: str) -> str:
    """
    Switches from *AaBb* into *aa_bb*.
    :param name: name to convert
    :return: converted name
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    return s2 if not keyword.iskeyword(s2) else s2 + "_"


def _process_example(code: str) -> str:
    """
    Add necessary imports to make the example work.
    """
    code = code.replace("", "")
    missing_imports = ["import numpy as np", "import onnx"]
    elements = [*missing_imports, "", "", code.strip("\n")]
    return "\n".join(elements)


def get_onnx_example(op_name, domain):
    """
    Retrieves examples associated to one operator
    stored in onnx packages.
    :param op_name: operator name
    :param domain: operator domain
    :param fmt: rendering format
    :return: dictionary
    """
    if domain in (None, "ai.onnx"):
        modules = [
            f"onnx.backend.test.case.node.{op_name.lower()}",
            f"onnx.backend.test.case.node.{pascal_to_snake_case(op_name)}",
        ]
    else:
        domain_ = domain.replace(".", "_")
        modules = [
            f"onnx.backend.test.case.node.{domain_}.{op_name.lower()}",
            f"onnx.backend.test.case.node.{domain_}.{pascal_to_snake_case(op_name)}",
        ]
    module = None
    for m in modules:
        try:
            mod = importlib.import_module(m)
            module = m
        except ImportError:
            continue
    if module is None:
        # Unable to find an example for 'op_name'.
        return {}
    results: dict[str, Any] = {}
    for v in mod.__dict__.values():
        if not isinstance(v, _Exporter):
            continue
        code_cls = inspect.getsource(v)
        codes = code_cls.split("@staticmethod")
        for me in v.__dict__:
            if not me.startswith("export"):
                continue
            sub = f" {me}()"
            found = None
            for code in codes:
                if sub in code:
                    found = code
            if found is None:
                raise RuntimeError(f"Unable to find {sub!r} in\n{code_cls}")
            found = textwrap.dedent(found)
            lines = found.split("\n")
            first = 0
            for i in range(len(lines)):
                if lines[i].startswith("def "):
                    first = i + 1
            found = textwrap.dedent("\n".join(lines[first:]))
            key = me[len("export") :]
            if key == "":
                key = "default"
                if key in results:
                    key = f"example {len(results) + 1}"
            results[key] = _process_example(found)
    return results


def is_last_schema(sch: OpSchema) -> bool:
    """
    Tells if this is the most recent schema for this operator.
    :param sch: schema
    :return: True
    """
    try:
        last = onnx.defs.get_schema(sch.name, domain=sch.domain)
    except onnx.defs.SchemaError:
        return True
    return last.since_version == sch.since_version


def onnx_documentation_folder(
    folder, title="ONNX Operators", flog=None, max_opsets=None
):
    """
    Creates documentation in a folder for all known
    ONNX operators or a subset.
    :param folder: folder where to write the documentation
    :param title: index title
    :param flog: logging function
    :param max_opsets: included operator definition up to this opsets
    :return: list of creates files
    """

    class _Table:
        def __init__(self, ops, domain, title=None):
            self.title = title or domain
            self.domain = domain
            self.ops = ops

        @property
        def domain_name(self):
            title = self.domain
            if title == "":
                title = "ai.onnx"
            return title

        def render(self, indent=""):
            table_dom = [""]
            table_dom.extend(
                [
                    ".. list-table::",
                    "    :widths: 10 10 10",
                    "    :header-rows: 1",
                    "",
                    "    * - operator",
                    "      - versions",
                    "      - differences",
                ]
            )

            for op in self.ops:
                name = op["name"]
                dom = self.domain.replace(".", "-")
                table_dom.append(f"    * - :ref:`{name} <l-onnx-doc{dom}-{name}>`")
                versions = sorted(
                    [(k, v) for k, v in op["links"].items() if isinstance(k, int)],
                    reverse=True,
                )
                col1 = ", ".join(f":ref:`{k} <{v}>`" for k, v in versions)
                diffs = sorted(
                    [(k, v) for k, v in op["links"].items() if isinstance(k, tuple)],
                    reverse=True,
                )
                col2 = ", ".join(f":ref:`{k[1]}/{k[0]} <{v}>`" for k, v in diffs)
                table_dom.append(f"      - {col1}")
                table_dom.append(f"      - {col2}")
            table_dom.append("")
            if indent != "":
                for i in range(len(table_dom)):
                    table_dom[i] = indent + table_dom[i]
            res = "\n".join(table_dom)
            return res

    all_schemas_available = _get_all_schemas_with_history()
    if len(all_schemas_available) < 3:
        raise RuntimeError(
            f"At least three domains are expected, found {list(all_schemas_available)}."
        )

    # filter out operator under development
    all_schemas = {}
    for domain, opset in all_schemas_available.items():
        max_version = None if max_opsets is None else max_opsets.get(domain, None)
        d = {}
        for op, schemas in opset.items():
            vers = {}
            for version, schema in schemas.items():
                if max_version is not None and version > max_version:
                    continue
                vers[version] = schema
            d[op] = vers
        all_schemas[domain] = d

    if len(all_schemas) < 3:
        raise RuntimeError(
            f"At leat three domains are expected, found {list(all_schemas)} in all_schemas."
        )

    if not os.path.exists(folder):
        os.makedirs(folder)

    pages = []
    tables = []

    # loop on domains
    for dom in sorted(all_schemas):
        sdom = "ai.onnx" if dom == "" else dom
        dom_pages = []

        do = all_schemas[dom]
        if len(do) == 0:
            raise RuntimeError(f"No operator for domain={dom!r}.")

        # loop on operators
        for op in sorted(do):
            if flog is not None:
                flog(f"generate page for onnx {dom!r} - {op!r}")
            page_name = f"onnx_{dom.replace('.', '')}_{op}"
            doc, d_links, n_examples = get_markdown_doc(
                folder, op, domain=dom, version=None, example=True, diff=True
            )
            if flog is not None and n_examples == 0:
                flog(f"{' '* 14}no_example for {op} from domain {domain}")
            if dom == "":
                main = op
            else:
                main = f"{dom} - {op}"
            sdom = dom.replace(".", "-")
            # Target in MyST https://myst-parser.readthedocs.io/en/v0.15.1/syntax/syntax.html?highlight=role#extra-markdown-syntax
            ref_link = f"(l-onnx-doc{sdom}-{op})="
            rows = [
                "",
                ref_link,
                "",
                f"# {main}",
                "",
                doc,
            ]

            full = os.path.join(folder, page_name + ".md")
            content = "\n".join(rows)
            pathlib.Path(full).write_text(content, encoding="utf-8")
            pages.append(full)
            dom_pages.append({"name": op, "links": d_links})

        tables.append(_Table(dom_pages, dom, sdom))

    # final
    if len(tables) < 3:
        raise RuntimeError(f"At least three domain are expected not {len(tables)}.")
    index = _template_main.render(pages=pages, tabs=tables, os=os, len=len, title=title)
    index = _clean_unicode(index)
    page_name = os.path.join(folder, "index.rst")
    pathlib.Path(page_name).write_text(index, encoding="utf-8")
    pages.append(page_name)
    return pages


def _generate_op_doc(app):
    logger = logging.getLogger(__name__)
    folder = app.config.onnx_doc_folder
    max_opsets = app.config.max_opsets
    onnx_documentation_folder(folder, flog=logger.info, max_opsets=max_opsets)


def _copy_repo_docs(app):
    logger = logging.getLogger(__name__)
    dest_name = app.config.onnx_md_folder

    docs_dir = pathlib.Path(__file__).parent.parent.parent  # docs
    dest_folder = docs_dir / "docsgen" / "source" / dest_name
    dest_folder.mkdir(parents=True, exist_ok=True)
    # Copy all the markdown files from the folder except for the blocklisted ones

    logger.info("Copying Markdown files from '%s' to '%s'", docs_dir, dest_folder)
    for file in docs_dir.glob("*.md"):
        if file.name in REPO_DOCS_EXCLUDE:
            continue
        shutil.copy(file, dest_folder)
        logger.info("Copying '%s'", file.name)


def setup(app):
    """
    Sphinx extension `onnx_sphinx` displays documentation
    on ONN Operators.
    """
    import sphinx

    app.add_config_value("onnx_doc_folder", "operators", "env")
    # Folder for storing the Markdown documentation from the repository
    app.add_config_value("onnx_md_folder", "repo-docs", "env")
    app.add_config_value("max_opsets", {}, "env")
    app.connect("builder-inited", _generate_op_doc)
    app.connect("builder-inited", _copy_repo_docs)
    return {"version": sphinx.__display_version__, "parallel_read_safe": True}


if "debug" in sys.argv:
    print("DEBUG")
    onnx_documentation_folder("_debug", flog=print)
    print("END")
