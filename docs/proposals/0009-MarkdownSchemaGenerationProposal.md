<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->

- Feature Name: `markdown_operator_schemas`
- Start Date: 2025-07-13
- RFC PR: [onnx/onnx#0000](https://github.com/onnx/onnx/pull/0000)
- Status: under discussion
- Authors:
  - cbourjau

# Proposal - Markdown Operator Schema Definition and Code Generation

## Summary
[summary]: #summary

Move the declarative part of each operator schema
(doc string, inputs, outputs, attributes, type constraints, static function bodies)
out of C++ into one Markdown file per operator version:
a TOML block followed by the doc string.
A file only states what changed relative to the previous version of the operator.
At build time, a Python generator turns these files into C++ functions that return an `OpSchema`.
Shape inference, data propagation and context-dependent function body builder functions remain hand-written C++,
but each file names the functions its schema uses.
The registrations and the operator set classes in `onnx/defs/operator_sets*.h` are generated as well.
The registry content and all public APIs are unchanged,
except that the location of a schema (`OpSchema::file()`, `OpSchema::line()`) refers to its spec file.
The proposed schemas are not considered stable.
Downstream consumers should continue to consume ONNX schemas through the existing APIs (for now).

## Motivation
[motivation]: #motivation

Schemas are currently defined by chains of `OpSchema` builder calls in `onnx/defs/<area>/defs.cc` and `old.cc`,
often via generator functions shared by several operators (e.g. `ConvOpSchemaGenerator`),
with doc strings partly in `onnx/defs/doc_strings.cc`.
Declarative data and C++ logic are interleaved.
As a result:

- ONNX's nature as a specification is obscured.
  The operator specifications are scattered across a complex C++ codebase.
- Changing a schema, e.g. improving the specification or adding a type to a constraint,
  requires editing C++ and often a shared generator,
  and reviewers must work out which schemas are affected.
- Contributors need to learn the builder API and file layout to propose a purely declarative change.

The proposed setup with one file per operator version
makes the place to edit and the scope of a change obvious for contributors and reviewers.

## Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

### Layout

```
operator-specifications/<domain>/v<N>/<op>.md
```

The files live in a top-level directory rather than under `onnx/defs`,
since they define the standard independently of the C++ implementation.

`<domain>/v<N>/<op>.md` exists if and only if the operator `<op>` in `<domain>` has a schema with `since_version == N`.
File names are the lowercase operator names,
so that case-insensitive file systems cannot cause conflicts.
Each file states the operator's `name` and `domain`, so it is self-contained;
the generator checks that both match the path.

### Format and inheritance

A file is a fenced TOML block (with a longer fence if the TOML itself contains one)
followed by the doc string in Markdown.
Each file inherits from the previous version of the same operator.
Except for `name` and `domain`, which are required in every file,
each top-level TOML key and the doc string are inherited independently:

- key absent: inherited unchanged;
- key present: replaces the inherited value completely (lists and maps are restated in full);
- doc string empty: inherited.

`shape_inference`, `data_propagation` and `function_body_builders` are not inherited either.
Every file MUST set `shape_inference`,
either to the name of a C++ function or to `false` if the schema has no inference function.
Every schema SHOULD have an inference function.
`data_propagation` and `function_body_builders` are optional;
if absent, the schema has no such function.

The common indentation of all doc string lines but the first is removed,
so that indented lines are not rendered as code blocks.

Example, LeakyRelu.
`ai.onnx/v1/leakyrelu.md` declares everything:

````markdown
```toml
name = "LeakyRelu"
domain = "ai.onnx"
shape_inference = false

[attributes.alpha]
type = "FLOAT"
desc = "Coefficient of leakage default to 0.01."
default = 0.01

[attributes.consumed_inputs]
type = "INTS"
desc = "legacy optimization attribute."

[[inputs]]
name = "X"
type = "T"
desc = "Input tensor"

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
````

`ai.onnx/v6/leakyrelu.md` removes `consumed_inputs`, rewords `alpha`,
marks `X` and `Y` differentiable, and adds shape inference.
Type constraints and doc string are inherited from v1:

````markdown
```toml
name = "LeakyRelu"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[attributes.alpha]
type = "FLOAT"
desc = "Coefficient of leakage."
default = 0.01

[[inputs]]
name = "X"
type = "T"
desc = "Input tensor"
differentiation = "differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor"
differentiation = "differentiable"
```
````

`ai.onnx/v16/leakyrelu.md` adds `tensor(bfloat16)` via the type alias `all_float_types_ir4`
(`tensor(bfloat16)`, `tensor(float16)`, `tensor(float)`, `tensor(double)`)
and a function body, and restates `shape_inference`.
Everything else is inherited from the resolved v6 schema:

````markdown
```toml
name = "LeakyRelu"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."

[function_bodies]
16 = """
{
  Alpha = Constant <value_float: float = @alpha>()
  ...
  Y = Where (XLessThanZero, AlphaMulX, X)
}
"""
```
````

The generator also emits the registration of every version,
so no C++ is written for a schema apart from the functions it names.

## Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

### Keys

`name` and `domain` identify the operator; the registration sets them in C++.
All other keys map one-to-one onto `OpSchema` builder calls, with the builder's defaults.
Files list the keys in this order:

| Key | Builder call |
|-----|--------------|
| `support_level`, `deprecated` | corresponding setters |
| `node_determinism`, `num_outputs`, `allow_unchecked_attributes` | `SetNodeDeterminism`, `NumOutputs`, `AllowUncheckedAttributes` |
| `shape_inference` | `TypeAndShapeInferenceFunction`; function name or `false`; required, not inherited |
| `data_propagation` | `PartialDataPropagationFunction`; function name; optional, not inherited |
| `function_body_builders` | `SetContextDependentFunctionBodyBuilder`; map `opset_version -> function name`; optional, not inherited |
| `attributes` | `Attr`; map `name -> {type, desc, required, default}` |
| `inputs`, `outputs` | `Input`/`Output`; list items `{name, type, desc, option, homogeneous, min_arity, differentiation}` |
| `type_constraints` | `TypeConstraint`; map `name -> {types, desc}` |
| `function_bodies` | `FunctionBody`; map `opset_version -> ONNX text` |

The order of attributes, inputs, outputs and type constraints matches `docs/Operators.md`.
Attributes, inputs, outputs and type constraints are written as (arrays of) tables,
e.g. `[attributes.alpha]` and `[[inputs]]`.

Function names are C++ names relative to the `onnx` namespace,
e.g. `propagateShapeAndTypeFromFirstInput` or `defs::nn::utils::ConvShapeInference_opset22`.
The functions are declared in `onnx/defs/<area>/utils.h` for names starting with `defs::<area>::utils::`,
and in `onnx/defs/shape_inference.h` or `onnx/defs/data_propagators.h` for unqualified names;
the generator includes these headers.
Whether a named function exists is checked by the compiler.

The keys of `function_bodies` and `function_body_builders` are the opset versions
of the operators used in the respective body,
not the `since_version` of the schema.
A caller requesting the function of a schema for an opset version of its domain,
e.g. the `opset_import` of a model,
receives the body with the highest key that is less than or equal to that version.

The public API does not expose the values of `node_determinism`, `num_outputs` and `allow_unchecked_attributes`
as set by the builder calls:
`NumOutputs` and `AllowUncheckedAttributes` have no getters,
and `GetNodeDeterminism()` derives a value from the attributes, the function body and other schemas if none was set.
The spec files only contain the values that are set explicitly.
Storing the derived node determinism instead would state a value nobody declared
and would not follow later changes of the schemas it is derived from.
For the migration, the values are exposed temporarily (see [Migration](#migration)).

`types` is either a list of type strings (e.g. `["tensor(float)"]`)
or a string naming one of the existing `OpSchema` type helpers (e.g. `"all_tensor_types_ir13"`).
The format is internal to the build and not a public interface.
The generator validates every file
(e.g. unknown keys, `required` together with `default`, defaults that do not match the attribute type)
and reports errors with the file path.

### Code generation

For each `(domain, operator)` the generator resolves the versions in ascending order
and emits one complete C++ function per version,
e.g. `OpSchema onnx::generated::ai_onnx::LeakyRelu_ver16()`, with one namespace per domain.
Inheritance exists only in the source files.
`ai.onnx.ml` sources are guarded by `ONNX_ML`.

### Build integration

The generator lives in `tools/operator_specs/`, so it is not part of the wheel.
CMake invokes it via `add_custom_command`, as for `gen_proto.py`:

- inputs collected with `file(GLOB_RECURSE ... CONFIGURE_DEPENDS)`;
- one generated translation unit per `(domain, operator)` plus one header of declarations,
  so builds parallelize and an edit recompiles little;
- outputs are only rewritten if their content changed, so unchanged units are not recompiled.

The generator also emits the operator set classes `OpSet_<domain>_ver<N>`
and the functions registering them, e.g. `RegisterOnnxOperatorSetSchema`, from the layout.
`onnx/defs/operator_sets*.h` keep their paths and include the generated headers.
Adding a version of an operator therefore requires no change to these headers.
The range of opset versions of a domain remains set by `DomainToVersionRange` in `onnx/defs/schema.h`.

The registrations (`ONNX_*_OPERATOR_SET_SCHEMA`) are emitted into the translation unit of each operator.
A `#line` directive before each registration
sets the location reported by `OpSchema::file()` and `OpSchema::line()` to the spec file.

### Migration

The implementation is a very large refactoring.
Most of it is done mechanically by Python scripts based on tree-sitter,
which can be replayed on a newer base fairly easily, e.g. with LLM assistance.
Such deterministic scripts make it less likely that information gets lost in translation.
The scripts are initially part of the PRs but are removed before merging.

The migration is split into two PRs.
The first moves the declarative data into the spec files;
the second moves the names of the functions and generates the registrations.
This document describes the state after the second PR.

#### First PR: declarative data

After this PR, the spec files contain the declarative data,
i.e., all keys except `shape_inference`, `data_propagation` and `function_body_builders`.
The generator emits the schema functions only.
The registrations and the operator set classes remain hand-written,
and the registrations set the remaining parts in C++:

```cpp
ONNX_OPERATOR_SET_SCHEMA(
    LeakyRelu,
    16,
    generated::ai_onnx::LeakyRelu_ver16().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));
```

1. **Expose.**
   A temporary commit exposes the values set by `SetNodeDeterminism`, `NumOutputs` and `AllowUncheckedAttributes`
   in `OpSchema` and its Python bindings,
   i.e., the node determinism without derivation and the set of allowed output counts.
2. **Generate.**
   Add the generator and the CMake integration described above,
   without the keys `shape_inference`, `data_propagation` and `function_body_builders`,
   the registrations and the operator set classes.
   Without spec files, it emits no schemas.
3. **Export.**
   A script reads the registry of the current build (`onnx.defs.get_all_schemas_with_history()`)
   and writes the files, omitting every section equal to that of the previous version.
   A type constraint whose types equal an alias is written as that alias.
   Doc strings are dedented as described in [Format and inheritance](#format-and-inheritance).
4. **Rewrite registrations.**
   A script replaces `OpSchema()` in every registration by `generated::<domain>::<Name>_ver<N>()`
   and removes the builder calls whose information is contained in the spec files,
   in registrations and in schema generators.
   It fails on any builder call that is neither removed nor one of
   `TypeAndShapeInferenceFunction`, `PartialDataPropagationFunction` and `SetContextDependentFunctionBodyBuilder`.
5. **Remove dead code.**
   A script removes schema generators, doc strings (including `doc_strings.{h,cc}`) and helpers left unused.
   The documentation is regenerated.
6. **Revert** the commit of step 1 after the verification.

Steps 4 and 5 are applied in one commit, so that every commit builds with warnings as errors.

#### Second PR: functions and registrations

1. **Name functions.**
   Inference, data propagation and context-dependent function body builder lambdas,
   in registrations and in generator functions such as `ConvOpSchemaGenerator`,
   are replaced by functions named `<Op>ShapeInference_opset<N>`, `<Op>DataPropagation_opset<N>` and `<Op>FunctionBody_opset<N>`.
   These and the existing named functions are moved to `onnx/defs/<area>/utils.{h,cc}`;
   existing functions keep their names, and identical functions are not merged.
   Functions of the public API remain as wrappers.
2. **Generate.**
   Add the keys `shape_inference`, `data_propagation` and `function_body_builders` to the generator,
   as well as the registrations and the operator set classes.
3. **Move the function names.**
   A script writes the names of the functions passed to the remaining builder calls to the spec files.
4. **Remove the hand-written registrations.**
   A script removes the registrations from `defs.cc`/`old.cc`
   and the operator set classes from `onnx/defs/operator_sets*.h`, which include the generated ones instead.
   It fails on any registration with builder calls not covered by the spec files.
   Helpers left unused are removed.
   `defs.cc`/`old.cc` are deleted, except for two files that define functions of the public API.

Steps 3 and 4 are applied in one commit,
since the generated registrations would otherwise duplicate the hand-written ones.

#### Verification

Each PR is verified mechanically:
a dump of all schemas in the registry, and the generated `docs/Operators.md` and `docs/Changelog.md`,
must be functionally equivalent before and after.
Differences such as whitespace in doc strings or the order in which a constraint lists its types are acceptable.
In the first PR, both dumps are taken from builds that include the commit of step 1,
so that they contain the values exposed by it.

Each PR switches all schemas at once.
Between the two PRs, the spec files hold the declarative data and the C++ registrations the functions,
so no value is defined in both places.

## Drawbacks
[drawbacks]: #drawbacks

- **Verbose tables.**
  TOML 1.0 does not allow newlines within inline tables,
  so each attribute, input, output and type constraint is written as a table of several lines (e.g. `[[inputs]]`)
  rather than as a compact multi-line inline table.
  TOML 1.1.0 lifts this restriction,
  but `tomllib` supports it only from Python 3.15 ([docs](https://docs.python.org/3/library/tomllib.html)).
- **Indirection.**
  Declarative data and C++ logic for one operator live in different files.
  In many cases, this is already so today due to generator functions.
- **Build dependencies.**
  The generator uses `pydantic` for validation and `jinja2` for templates.
  Both become build-time requirements of source builds.

## Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

- **Inference functions are not inherited.**
  Shape inference often changes between versions of an operator.
  Requiring `shape_inference` in every file, with an explicit `false` for schemas without one,
  makes the function each schema uses visible in its file and in diffs,
  and prevents a new version from silently reusing an outdated function.
  TOML has no null value, hence `false`.
- **Whole-section replacement** instead of merging within sections
  keeps each section readable in one place and the order of inputs and outputs explicit.
- **TOML instead of YAML.**
  TOML requires quoted strings and has no implicit typing or null value,
  which leaves fewer ways to write a value ambiguously.
  It is parsed by the standard library's `tomllib`.
- **`pydantic` and `jinja2`** keep the generator short and readable;
  this is preferred over avoiding build dependencies.
- **Markdown with a TOML block** instead of a pure TOML file
  keeps long doc strings editable and rendered on GitHub.
- **Fenced block instead of frontmatter.**
  GitHub renders YAML frontmatter as a horizontal table
  ([announcement](https://github.blog/news-insights/product-news/viewing-yaml-metadata-in-your-documents/)),
  which suits flat metadata but not nested lists and maps such as `inputs` or `attributes`.
  A fenced block is displayed verbatim with syntax highlighting.
  Generic frontmatter parsers do not recognize it,
  but the files are only read by the ONNX generator.

## Unresolved questions
[unresolved-questions]: #unresolved-questions

None currently.

## Future possibilities
[future-possibilities]: #future-possibilities

- Generate (or retire) `docs/Operators.md` and `docs/Changelog.md` in favor of just the schema files.
- Render the operator documentation on [onnx.ai](https://onnx.ai/) from the schema files directly.
- Stabilize the schema to let external tools read operator definitions directly from the files.
