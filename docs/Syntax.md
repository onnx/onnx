<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

Overview
========

This document describes a textual syntax for ONNX models, which is currently an experimental feature.
The syntax enables a compact and readable representation of ONNX models. It is motivated by a couple
of use-cases. One is to enable compact description of test-cases and its use in CI (both in the ONNX
repo as well as in other dependent repos such as ONNX-MLIR). The second is to help simplify the
definition of ONNX functions. Several of the existing function-definitions are verbose, and the
use of this syntax will lead to more compact, readable, and easier-to-maintain function definitions.
Efficient representation and efficient parsing of very large tensor-constants is *not* a goal.
Alternative methods should be used for that.

The API
-------

The key parser methods are the ```OnnxParser::Parse``` methods, used as below.

```cpp
  const char* code = R"ONNX(
<
  ir_version: 7,
  opset_import: [ "" : 10 ]
>
agraph (float[N, 128] X, float[128, 10] W, float[10] B) => (float[N, 10] C)
{
    T = MatMul(X, W)
    S = Add(T, B)
    C = Softmax(S)
}
)ONNX";

  ModelProto model;
  OnnxParser::Parse(model, code);

  checker::check_model(model);
```

See the [test-cases](../onnx/test/cpp/parser_test.cc) for more examples illustrating the API and syntax.

The Syntax
----------

The grammar below describes the syntax:

```
   id-list ::= id (',' id)*
   tensor-dim ::= '?' | id | int-constant
   tensor-dims ::= tensor-dim (',' tensor-dim)*
   tensor-type ::= prim-type | prim-type '[' ']' | prim-type '[' tensor-dims ']'
   type ::= tensor-type | 'seq' '(' type ')' | 'map' '(' prim-type ',' type ')'
            | 'optional' '(' type ')' | 'sparse_tensor' '(' tensor-type ')'
   value-info ::= type id
   value-infos ::= value-info (',' value-info)*
   value-info-list ::= '(' value-infos? ')'
   prim-constants ::= prim-constant (',' prim-constant)*
   tensor-constant ::= tensor-type (id)? ('=')? '{' prim-constants '}'
   attr-ref ::= '@' id
   single-attr-value ::= tensor-constant | graph | prim-constant | attr-ref
   attr-value-list ::= '[' single-attr-value (',' single-attr-value)* ']'
   attr-value ::= single-attr-value | attr-value-list
   attr-type ::= ':' id
   attr ::= id attr-type? '=' attr-value
   attr-list ::= '<' attr (',' attr)* '>'
   node ::= id-list? '=' qualified-id attr-list? '(' id-list? ')'
         |  id-list? '=' qualified-id '(' id-list? ')' attr-list
   node-list ::= '{' node* '}'
   graph ::= id value-info-list '=>' value-info-list node-list
   other-data ::= id ':' value
   other-data-list ::= '<' other-data (',' other-data)* '>'
   fun-attr-list ::= '<' id | attr (',' id | attr)* '>'
   fun-input-list ::= '(' id-list ')'
   fun-output-list ::= '(' id-list ')'
   function ::= other-data-list? id fun-attr-list?  fun-input-list '=>' fun-output-list  node-list
   model ::= other-data-list? graph function*
```
