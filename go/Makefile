all: onnx.pb.go

$(GOPATH)/bin/protoc-gen-go:
	go get -u github.com/golang/protobuf/protoc-gen-go

onnx.pb.go: $(GOPATH)/bin/protoc-gen-go ../onnx/onnx.proto
	protoc --go_out=. --proto_path=../onnx ../onnx/onnx.proto
