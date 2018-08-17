#include "onnx/onnxifi_loader.h"
#include "test_drive.h"

int main(){
	string location = "Write Me";
	onnxifi_library dummy_backend;
	if (!onnxifi_load(1, NULL, &dummy_backend)){
		//Backend loading failed
	}
	return 0;
}
