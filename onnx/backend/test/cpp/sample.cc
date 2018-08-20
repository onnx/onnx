#include "onnx/onnxifi_loader.h"
#include "driver/test_driver.h"

int main(){
	std::string location = "Write Me";
	onnxifi_library dummy_backend;
	if (!onnxifi_load(1, NULL, &dummy_backend)){
		//Backend loading failed
	}
	return 0;
}
