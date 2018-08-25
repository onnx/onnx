This directory contains the binding for the Go language.

The package is go-gettable

`go get github.com/onnx/onnx/go`

Then import it as `onnx`

```go
import onnx "github.com/onnx/onnx/go
```

# How to generate the file

You should not need to re-generate this file. But if you want/need to tweek it, here is what you need:

* a regular `protoc` installation; 
* the [gogoprotobuf](https://github.com/gogo/protobuf/) utilities;
* a Go toolchain

a Simple makefile is  provided to generate the file for you.
Simply run `make`


