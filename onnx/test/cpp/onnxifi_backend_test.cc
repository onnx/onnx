#include <iostream>

#include "gtest/gtest.h"
#include "onnx/onnxifi_loader.h"
#include "onnx/onnxifi.h"

#if defined(__APPLE__)
#define ONNXIFI_DUMMY_LIBRARY "libonnxifi_dummy.dylib"
#elif defined(_WIN32)
#define ONNXIFI_DUMMY_LIBRARY L"onnxifi_dummy.dll"
#else
#define ONNXIFI_DUMMY_LIBRARY "libonnxifi_dummy.so"
#endif


namespace ONNX_NAMESPACE
{
	namespace Test
	{
		TEST(OnnxifiLoadTest, OnnxifiDummyBackend)
		{

			#define EXPECT_EQ_OSS(X) EXPECT_EQ(X, ONNXIFI_STATUS_SUCCESS)
			onnxifi_library dummy_backend;
			EXPECT_TRUE(onnxifi_load(1, ONNXIFI_DUMMY_LIBRARY,  &dummy_backend));

			onnxBackendID backendID;
			onnxBackend backend;
			onnxEvent event;
			onnxGraph graph;

			//Testing onnxGetBackendIDs
			size_t numBackends = -1;

                        EXPECT_EQ_OSS(dummy_backend.onnxGetBackendIDs(
                            &backendID, &numBackends));
                        EXPECT_EQ(numBackends, 1);

                        //Testing onnxReleaseBackendID
                        EXPECT_EQ_OSS(
                            dummy_backend.onnxReleaseBackendID(backendID));

                        // Testing onnxGetBackendInfo
                        onnxBackendInfo infoType = 0;
			char infoValue[11] = "abc";
			size_t infoValueSize = 3;

                        EXPECT_EQ_OSS(dummy_backend.onnxGetBackendInfo(
                            backendID, infoType, infoValue, &infoValueSize));
                        EXPECT_EQ(infoValue[0], 0);

                        //Testing onnxGetBackendCompatibility
			char onnxModel[] = "";
			size_t onnxModelSize = 0;
			EXPECT_EQ_OSS(dummy_backend.onnxGetBackendCompatibility(backendID, onnxModelSize, onnxModel));

			//Testing onnxInitBackend
			uint64_t auxpropertiesList = 0;
			EXPECT_EQ_OSS(dummy_backend.onnxInitBackend(backendID, &auxpropertiesList, &backend));

			//Testing onnxReleaseBackend
			EXPECT_EQ_OSS(dummy_backend.onnxReleaseBackend(backend));

			//Testing onnxInitEvent
			EXPECT_EQ_OSS(dummy_backend.onnxInitEvent(backend, &event));

			//Testing onnxSignalEvent
			EXPECT_EQ_OSS(dummy_backend.onnxSignalEvent(event));

			//Testing onnxWaitEvent
			EXPECT_EQ_OSS(dummy_backend.onnxWaitEvent(event));

			//Testing onnxReleaseEvent
			EXPECT_EQ_OSS(dummy_backend.onnxReleaseEvent(event));

			//Testing onnxInitGraph
			uint32_t weightCount = 1;
                        onnxTensorDescriptorV1 weightDescriptors;
                        EXPECT_EQ_OSS(dummy_backend.onnxInitGraph(backend, onnxModelSize, &onnxModel, weightCount, &weightDescriptors, &graph));

			//Testing onnxInitGraph
			uint32_t inputsCount = 1;
                        onnxTensorDescriptorV1 inputDescriptors;
                        uint32_t outputsCount = 1;
                        onnxTensorDescriptorV1 outputDescriptors;
                        EXPECT_EQ_OSS(dummy_backend.onnxSetGraphIO(graph, inputsCount, &inputDescriptors, outputsCount, &outputDescriptors));

			//Testing onnxRunGraph
                        onnxMemoryFenceV1 inputFence, outputFence;
                        EXPECT_EQ_OSS(dummy_backend.onnxRunGraph(graph, &inputFence, &outputFence));
			EXPECT_EQ(outputFence.type, ONNXIFI_SYNCHRONIZATION_EVENT);

			//Testing onnxReleaseGraph
			EXPECT_EQ_OSS(dummy_backend.onnxReleaseGraph(graph));

#undef EXPECT_EQ_OSS
                }

	}
}
