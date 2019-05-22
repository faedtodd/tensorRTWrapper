#include "TrtNet.h"
#include "EntroyCalibrator.h"
#include <cassert>
#include <chrono>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <time.h>
#include <unordered_map>

using namespace std;
using namespace nvinfer1;
//using namespace nvcaffeparser1;
using namespace nvonnxparser;
//using namespace plugin;

static Tn::Logger gLogger;

inline void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

namespace Tn
{
    trtNet::trtNet(const std::string& onnxmodel, int maxBatchSize)
    :mTrtContext(nullptr),mTrtEngine(nullptr),mTrtRunTime(nullptr),mTrtRunMode(mode),mTrtInputCount(0),mTrtIterationTime(0),mTrtBatchSize(maxBatchSize)
    {
        IHostMemory* trtModelStream{nullptr};

        nvonnxparser::IPluginFactory* onnxPlugin = createPluginFactory(gLogger);

        ICudaEngine* tmpEngine = loadModelAndCreateEngine(caffemodel.c_str(), maxBatchSize, parser, trtModelStream);
        assert(tmpEngine != nullptr);
        assert(trtModelStream != nullptr);

        tmpEngine->destroy();

        mTrtRunTime = createInferRuntime(gLogger);     
        assert(mTrtRunTime != nullptr);
        mTrtEngine= mTrtRunTime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), onnxPlugin);
        assert(mTrtEngine != nullptr);
        // Deserialize the engine.
        trtModelStream->destroy();

        InitEngine();
    }

    trtNet::trtNet(const std::string& engineFile)
    :mTrtContext(nullptr),mTrtEngine(nullptr),mTrtRunTime(nullptr),mTrtRunMode(RUN_MODE::FLOAT32),mTrtInputCount(0),mTrtIterationTime(0)
    {
	using namespace std;
	fstream file;
	cout << "loading filename from:" << engineFile << endl;
	nvonnxparser::IPluginFactory* onnxPlugin = createPluginFactory(gLogger);
	file.open(engineFile, ios::binary | ios::in);
	file.seekg(0, ios::end);
	int length = file.tellg();
	//cout << "length:" << length << endl;
	file.seekg(0, ios::beg);
	std::unique_ptr<char[]> data(new char[length]);
	file.read(data.get(), length);
	file.close();
	cout << "load engine done" << endl;

        std::cout << "deserializing" << std::endl;
        mTrtRunTime = createInferRuntime(gLogger);
        assert(mTrtRunTime != nullptr);
        mTrtEngine= mTrtRunTime->deserializeCudaEngine(data.get(), length, onnxPlugin);
        assert(mTrtEngine != nullptr);

        InitEngine();
    }

    void trtNet::InitEngine()
    {
        mTrtBatchSize = mTrtEngine->getMaxBatchSize();
        mTrtContext = mTrtEngine->createExecutionContext();
        assert(mTrtContext != nullptr);

        // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings()
        int nbBindings = mTrtEngine->getNbBindings();

        mTrtCudaBuffer.resize(nbBindings);
        mTrtBindBufferSize.resize(nbBindings);
        for (int i = 0; i < nbBindings; ++i)
        {
            Dims dims = mTrtEngine->getBindingDimensions(i);
            DataType dtype = mTrtEngine->getBindingDataType(i);
            int64_t totalSize = volume(dims) * mTrtBatchSize * getElementSize(dtype);
            mTrtBindBufferSize[i] = totalSize;
            mTrtCudaBuffer[i] = safeCudaMalloc(totalSize);
            if(mTrtEngine->bindingIsInput(i))
                mTrtInputCount++;
        }

        CUDA_CHECK(cudaStreamCreate(&mTrtCudaStream));
    }


    nvinfer1::ICudaEngine* trtNet::loadModelAndCreateEngine(const char* modelFile,int maxBatchSize, IHostMemory*& trtModelStream)
    {
        // Create the builder
        IBuilder* builder = createInferBuilder(gLogger);
        assert(builder != nullptr);

        // Parse the model to populate the network
        nvinfer1::INetworkDefinition* network = builder->createNetwork();
        auto parser = nvonnxparser::createParser(*network, gLogger);

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 30);
	builder->setFp16Mode(true);
	builder->setInt8Mode(false);

	samplesCommon::enableDLA(builder, gArgs.useDLACore);
	cout << "start building engine" << endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	cout << "build engine done" << endl;
	assert(engine);

        // We don't need the network any more, and we can destroy the parser.
        network->destroy();
        parser->destroy();

        // Serialize the engine, then close everything down.
        trtModelStream = engine->serialize();

        builder->destroy();
        return engine;
    }

    void trtNet::doInference(const void* inputData, void* outputData ,int batchSize)
    {
        //static const int batchSize = 1;
        assert(mTrtInputCount == 1);
        assert(batchSize <= mTrtBatchSize);

        // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
        int inputIndex = 0;
        CUDA_CHECK(cudaMemcpyAsync(mTrtCudaBuffer[inputIndex], inputData, mTrtBindBufferSize[inputIndex], cudaMemcpyHostToDevice, mTrtCudaStream));
        auto t_start = std::chrono::high_resolution_clock::now();
        mTrtContext->execute(batchSize, &mTrtCudaBuffer[inputIndex]);
        auto t_end = std::chrono::high_resolution_clock::now();
        float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        std::cout << "Time taken for inference is " << total << " ms." << std::endl;

        for (size_t bindingIdx = mTrtInputCount; bindingIdx < mTrtBindBufferSize.size(); ++bindingIdx)
        {
            auto size = mTrtBindBufferSize[bindingIdx];
            CUDA_CHECK(cudaMemcpyAsync(outputData, mTrtCudaBuffer[bindingIdx], size, cudaMemcpyDeviceToHost, mTrtCudaStream));
            outputData = (char *)outputData + size;
        }

        //cudaStreamSynchronize(mTrtCudaStream);

        mTrtIterationTime ++ ;
    }
}
