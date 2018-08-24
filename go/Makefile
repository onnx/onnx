all: onnx.pb.go

$(GOPATH)/bin/protoc-gen-gofast: $(GOPATH)/bin/protoc-gen-go
	go get github.com/gogo/protobuf/protoc-gen-gofast

$(GOPATH)/bin/protoc-gen-go:
	go get -u github.com/golang/protobuf/protoc-gen-go

onnx.pb.go: $(GOPATH)/bin/protoc-gen-gofast ../onnx/onnx.proto
	protoc --gofast_out=. --proto_path=../onnx ../onnx/onnx.proto
