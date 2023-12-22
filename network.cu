#include "network.h"

__global__ void calculateDiffTensor(DATATYPE* result, DATATYPE* first, DATATYPE* second, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    result[idx] = first[idx] - second[idx];
}

Network::Network(
    CNN_t		    _cnnType,
    vdnn_t		    _vdnnType,
    vdnnAlgoMode_t	_algoMode,
    int		        _batchSize) 
{
    cudaFree(0);
    cnnType = _cnnType;
    vdnnType = _vdnnType;
    algoMode = _algoMode;
    batchSize = _batchSize;
    numLayers = 0;
    inputData = NULL;
    labelData = NULL;
    networkDiffData = NULL;
    networkGradData = NULL;
    workSpace = NULL;
    for (int i = 0; i < DEFAULT_NUM_LAYER; i++) {
        layer[i] = NULL;
        isOffload[i] = false;
        isPrefetched[i] = false;
        offloadedSrcData_h[i] = NULL;
    }

    /* Get CUDA device properties */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, DEFAULT_DEVICE);

    /* Setup cudnnHandle, cublasHandle, and two CUDA streams */
    checkCUDNN(cudnnCreate(&cudnnHandle));
    checkCUBLAS(cublasCreate(&cublasHandle));
    checkCudaErrors(cudaStreamCreate(&stream_compute));
    checkCudaErrors(cudaStreamCreate(&stream_memory));
    cudaStream_t streamArray[2] = {stream_compute, stream_memory};
    size_t streamSizes[2] = {0, 0};

    /* Initialize cnmem: notice how only one device is used (TITAN Xp) */
    cnmemDevice_t cnmemDevice;
    cnmemDevice.device = DEFAULT_DEVICE;
    cnmemDevice.size = (prop.totalGlobalMem / (size_t)1e9) * (size_t)1e9;
    cnmemDevice.numStreams = 2;
    cnmemDevice.streams = streamArray;
    cnmemDevice.streamSizes = streamSizes;
    ASSERT_EQ(cnmemInit(1, &cnmemDevice, 0));

    /* Set cuDNN and cuBLAS stream to stream_compute */
    cudnnSetStream(cudnnHandle, stream_compute);
    cublasSetStream(cublasHandle, stream_compute);

    /* If VDNN_DYNAMIC_ALGO, initialize dynamically */
    if (_algoMode == VDNN_DYNAMIC_ALGO) {
        /* Temporarily set to MEMORY_OPT */
        algoMode = VDNN_MEMORY_OPT_ALGO;

        /* Set layers */
        setLayers(_cnnType);

        /* Set to VDNN_ALL */
        setVdnnType(VDNN_ALL);

        /* Initialize input data */
        ASSERT_EQ(cnmemMalloc(&inputData, sizeof(DATATYPE) * getInputSize(), NULL));

        /* Initialize label data */
        ASSERT_EQ(cnmemMalloc(&labelData, sizeof(DATATYPE) * getOutputSize(), NULL));

        /* VDNN_DYNAMIC_ALGO */
        vdnnDynamicAlgo();
    }
    else {
        /* Initialize layers of network */
        setLayers(_cnnType);

        /* Set host memory, dx/dy data, and workspace */
        setVdnnType(_vdnnType);

        /* Initialize input data */
        ASSERT_EQ(cnmemMalloc(&inputData, sizeof(DATATYPE) * getInputSize(), NULL));

        /* Initialize label data */
        ASSERT_EQ(cnmemMalloc(&labelData, sizeof(DATATYPE) * getOutputSize(), NULL));
    }
}

Network::~Network() {
    cudnnDestroy(cudnnHandle);
    cublasDestroy(cublasHandle);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_memory);
}

void Network::reset(CNN_t _cnnType, vdnn_t _vdnnType, vdnnAlgoMode_t _algoMode) {
    /* Release all allocated memory */
    cnmemFinalize();

    /* Some declarations */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, DEFAULT_DEVICE);
    cudaStream_t streamArray[2] = {stream_compute, stream_memory};
    size_t streamSizes[2] = {0, 0};

    /* Initialize cnmem: notice how only one device is used (TITAN Xp) */
    cnmemDevice_t cnmemDevice;
    cnmemDevice.device = DEFAULT_DEVICE;
    cnmemDevice.size = (prop.totalGlobalMem / (size_t)1e9) * (size_t)1e9;
    cnmemDevice.numStreams = 2;
    cnmemDevice.streams = streamArray;
    cnmemDevice.streamSizes = streamSizes;
    ASSERT_EQ(cnmemInit(1, &cnmemDevice, 0));

    /* Initialize layers of network */
    algoMode = _algoMode;
    setLayers(_cnnType);

    /* Set host memory, dx/dy data, and workspace */
    setVdnnType(_vdnnType);

    /* Initialize input data */
    ASSERT_EQ(cnmemMalloc(&inputData, sizeof(DATATYPE) * getInputSize(), NULL));

    /* Initialize label data */
    ASSERT_EQ(cnmemMalloc(&labelData, sizeof(DATATYPE) * getOutputSize(), NULL));
}

void Network::setLayers(CNN_t _cnnType) {
    cnnType = _cnnType;
    if (_cnnType == ALEXNET)
        setLayersAlexnet();
    else if (_cnnType == OVERFEAT)
        setLayersOverfeat();
    else if (_cnnType == GOOGLENET)
        setLayersGooglenet();
    else if (_cnnType == VGG16_64)
        setLayersVgg16(64);
    else if (_cnnType == VGG16_128)
        setLayersVgg16(128);
    else if (_cnnType == VGG16_256)
        setLayersVgg16(256);
    else if (_cnnType == CUSTOM_CNN)
        setLayersCustomCNN();
    else
        M_ASSERT(false, "Invalid CNN Type!");
    printf("-------------------------------------------------------\n\n");
    sanityCheck();
}

void Network::setLayersAlexnet() {
    numLayers = 19;

    /* Set input dimensions */
    n_in = batchSize;
    c_in = 3;
    h_in = 227;
    w_in = 227;

    /* Set output dimensions */
    n_out = batchSize;
    c_out = 1000;
    h_out = 1;
    w_out = 1;

    layer[0] = (Layer*) malloc(sizeof(Layer));
    *layer[0] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 0, 
        batchSize, 3, 227, 227, 
        0, 0, 4, 4, 
        96, 11, 11
    );
    layer[1] = (Layer*) malloc(sizeof(Layer));
    *layer[1] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 1, 
        batchSize, 96, 55, 55,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[2] = (Layer*) malloc(sizeof(Layer));
    *layer[2] = Layer(
        vdnnType, algoMode, POOL, 
        &cudnnHandle, &stream_compute, &stream_memory, 2, 
        batchSize, 96, 55, 55, 
        0, 0, 2, 2, 
        0, 3, 3
    );
    layer[3] = (Layer*) malloc(sizeof(Layer));
    *layer[3] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 3, 
        batchSize, 96, 27, 27, 
        2, 2, 1, 1, 
        256, 5, 5
    );
    layer[4] = (Layer*) malloc(sizeof(Layer));
    *layer[4] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 4, 
        batchSize, 256, 27, 27,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[5] = (Layer*) malloc(sizeof(Layer));
    *layer[5] = Layer(
        vdnnType, algoMode, POOL, 
        &cudnnHandle, &stream_compute, &stream_memory, 5, 
        batchSize, 256, 27, 27, 
        0, 0, 2, 2, 
        0, 3, 3
    );
    layer[6] = (Layer*) malloc(sizeof(Layer));
    *layer[6] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 6, 
        batchSize, 256, 13, 13, 
        1, 1, 1, 1, 
        384, 3, 3
    );
    layer[7] = (Layer*) malloc(sizeof(Layer));
    *layer[7] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 7, 
        batchSize, 384, 13, 13,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[8] = (Layer*) malloc(sizeof(Layer));
    *layer[8] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 8, 
        batchSize, 384, 13, 13, 
        1, 1, 1, 1, 
        384, 3, 3
    );
    layer[9] = (Layer*) malloc(sizeof(Layer));
    *layer[9] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 9, 
        batchSize, 384, 13, 13,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[10] = (Layer*) malloc(sizeof(Layer));
    *layer[10] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 10, 
        batchSize, 384, 13, 13, 
        1, 1, 1, 1, 
        256, 3, 3
    );
    layer[11] = (Layer*) malloc(sizeof(Layer));
    *layer[11] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 11, 
        batchSize, 256, 13, 13,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[12] = (Layer*) malloc(sizeof(Layer));
    *layer[12] = Layer(
        vdnnType, algoMode, POOL, 
        &cudnnHandle, &stream_compute, &stream_memory, 12, 
        batchSize, 256, 13, 13, 
        0, 0, 2, 2, 
        0, 3, 3
    );
    layer[13] = (Layer*) malloc(sizeof(Layer));
    *layer[13] = Layer(
        vdnnType, algoMode, FC_GEMM, 
        &cublasHandle, &stream_compute, &stream_memory, 13, 
        batchSize, 9216, 1, 1,
        0, 0, 0, 0,
        4096, 1, 1
    );
    layer[14] = (Layer*) malloc(sizeof(Layer));
    *layer[14] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 14, 
        batchSize, 4096, 1, 1,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[15] = (Layer*) malloc(sizeof(Layer));
    *layer[15] = Layer(
        vdnnType, algoMode, FC_GEMM, 
        &cublasHandle, &stream_compute, &stream_memory, 15, 
        batchSize, 4096, 1, 1,
        0, 0, 0, 0,
        4096, 1, 1
    );
    layer[16] = (Layer*) malloc(sizeof(Layer));
    *layer[16] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 16, 
        batchSize, 4096, 1, 1,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[17] = (Layer*) malloc(sizeof(Layer));
    *layer[17] = Layer(
        vdnnType, algoMode, FC_GEMM, 
        &cublasHandle, &stream_compute, &stream_memory, 17, 
        batchSize, 4096, 1, 1,
        0, 0, 0, 0,
        1000, 1, 1
    );
    layer[18] = (Layer*) malloc(sizeof(Layer));
    *layer[18] = Layer(
        vdnnType, algoMode, SOFTMAX, 
        &cudnnHandle, &stream_compute, &stream_memory, 18, 
        batchSize, 1000, 1, 1,
        0, 0, 0, 0,
        0, 0, 0
    );
}

void Network::setLayersOverfeat() {
    /* Note: for layers 13, 15, and 17, VDNN_MEMORY_OPT_ALGO 
    option is used because these layers are intended to be 
    FC_GEMM, but simply implemented as FC_CONV */
    numLayers = 19;

    /* Set input dimensions */
    n_in = batchSize;
    c_in = 3;
    h_in = 231;
    w_in = 231;

    /* Set output dimensions */
    n_out = batchSize;
    c_out = 1000;
    h_out = 1;
    w_out = 1;

    layer[0] = (Layer*) malloc(sizeof(Layer));
    *layer[0] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 0, 
        batchSize, 3, 231, 231, 
        0, 0, 4, 4, 
        96, 11, 11
    );
    layer[1] = (Layer*) malloc(sizeof(Layer));
    *layer[1] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 1, 
        batchSize, 96, 56, 56,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[2] = (Layer*) malloc(sizeof(Layer));
    *layer[2] = Layer(
        vdnnType, algoMode, POOL, 
        &cudnnHandle, &stream_compute, &stream_memory, 2, 
        batchSize, 96, 56, 56, 
        0, 0, 2, 2, 
        0, 2, 2
    );
    layer[3] = (Layer*) malloc(sizeof(Layer));
    *layer[3] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 3, 
        batchSize, 96, 28, 28, 
        0, 0, 1, 1, 
        256, 5, 5
    );
    layer[4] = (Layer*) malloc(sizeof(Layer));
    *layer[4] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 4, 
        batchSize, 256, 24, 24,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[5] = (Layer*) malloc(sizeof(Layer));
    *layer[5] = Layer(
        vdnnType, algoMode, POOL, 
        &cudnnHandle, &stream_compute, &stream_memory, 5, 
        batchSize, 256, 24, 24, 
        0, 0, 2, 2, 
        0, 2, 2
    );
    layer[6] = (Layer*) malloc(sizeof(Layer));
    *layer[6] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 6, 
        batchSize, 256, 12, 12, 
        1, 1, 1, 1, 
        512, 3, 3
    );
    layer[7] = (Layer*) malloc(sizeof(Layer));
    *layer[7] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 7, 
        batchSize, 512, 12, 12,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[8] = (Layer*) malloc(sizeof(Layer));
    *layer[8] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 8, 
        batchSize, 512, 12, 12, 
        1, 1, 1, 1, 
        1024, 3, 3
    );
    layer[9] = (Layer*) malloc(sizeof(Layer));
    *layer[9] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 9, 
        batchSize, 1024, 12, 12,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[10] = (Layer*) malloc(sizeof(Layer));
    *layer[10] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 10, 
        batchSize, 1024, 12, 12, 
        1, 1, 1, 1, 
        1024, 3, 3
    );
    layer[11] = (Layer*) malloc(sizeof(Layer));
    *layer[11] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 11, 
        batchSize, 1024, 12, 12,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[12] = (Layer*) malloc(sizeof(Layer));
    *layer[12] = Layer(
        vdnnType, algoMode, POOL, 
        &cudnnHandle, &stream_compute, &stream_memory, 12, 
        batchSize, 1024, 12, 12, 
        0, 0, 2, 2, 
        0, 2, 2
    );
    layer[13] = (Layer*) malloc(sizeof(Layer));
    *layer[13] = Layer(
        vdnnType, VDNN_MEMORY_OPT_ALGO, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 13, 
        batchSize, 1024, 6, 6,
        0, 0, 1, 1,
        3072, 6, 6
    );
    layer[14] = (Layer*) malloc(sizeof(Layer));
    *layer[14] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 14, 
        batchSize, 3072, 1, 1,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[15] = (Layer*) malloc(sizeof(Layer));
    *layer[15] = Layer(
        vdnnType, VDNN_MEMORY_OPT_ALGO, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 15, 
        batchSize, 3072, 1, 1,
        0, 0, 1, 1,
        4096, 1, 1
    );
    layer[16] = (Layer*) malloc(sizeof(Layer));
    *layer[16] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 16, 
        batchSize, 4096, 1, 1,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[17] = (Layer*) malloc(sizeof(Layer));
    *layer[17] = Layer(
        vdnnType, VDNN_MEMORY_OPT_ALGO, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 17, 
        batchSize, 4096, 1, 1,
        0, 0, 1, 1,
        1000, 1, 1
    );
    layer[18] = (Layer*) malloc(sizeof(Layer));
    *layer[18] = Layer(
        vdnnType, algoMode, SOFTMAX, 
        &cudnnHandle, &stream_compute, &stream_memory, 18, 
        batchSize, 1000, 1, 1,
        0, 0, 0, 0,
        0, 0, 0
    );
}

void Network::setLayersGooglenet() {
    // TODO
}

void Network::setLayersVgg16(int givenBatch) {
    batchSize = givenBatch;
    numLayers = 37;

    /* Set input dimensions */
    n_in = batchSize;
    c_in = 3;
    h_in = 224;
    w_in = 224;

    /* Set output dimensions */
    n_out = batchSize;
    c_out = 1000;
    h_out = 1;
    w_out = 1;

    layer[0] = (Layer*) malloc(sizeof(Layer)); // conv1_1
    *layer[0] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 0, 
        batchSize, 3, 224, 224, 
        1, 1, 1, 1, 
        64, 3, 3
    );
    layer[1] = (Layer*) malloc(sizeof(Layer));
    *layer[1] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 1, 
        batchSize, 64, 224, 224,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[2] = (Layer*) malloc(sizeof(Layer)); // conv1_2
    *layer[2] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 2, 
        batchSize, 64, 224, 224, 
        1, 1, 1, 1, 
        64, 3, 3
    );
    layer[3] = (Layer*) malloc(sizeof(Layer));
    *layer[3] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 3, 
        batchSize, 64, 224, 224,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[4] = (Layer*) malloc(sizeof(Layer)); // pooling
    *layer[4] = Layer(
        vdnnType, algoMode, POOL, 
        &cudnnHandle, &stream_compute, &stream_memory, 4, 
        batchSize, 64, 224, 224, 
        0, 0, 2, 2, 
        0, 2, 2
    );
    layer[5] = (Layer*) malloc(sizeof(Layer)); // conv2_1
    *layer[5] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 5, 
        batchSize, 64, 112, 112, 
        1, 1, 1, 1, 
        128, 3, 3
    );
    layer[6] = (Layer*) malloc(sizeof(Layer));
    *layer[6] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 6, 
        batchSize, 128, 112, 112,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[7] = (Layer*) malloc(sizeof(Layer)); // conv2_2
    *layer[7] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 7, 
        batchSize, 128, 112, 112, 
        1, 1, 1, 1, 
        128, 3, 3
    );
    layer[8] = (Layer*) malloc(sizeof(Layer));
    *layer[8] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 8, 
        batchSize, 128, 112, 112,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[9] = (Layer*) malloc(sizeof(Layer)); // pooling
    *layer[9] = Layer(
        vdnnType, algoMode, POOL, 
        &cudnnHandle, &stream_compute, &stream_memory, 9, 
        batchSize, 128, 112, 112, 
        0, 0, 2, 2, 
        0, 2, 2
    );
    layer[10] = (Layer*) malloc(sizeof(Layer)); // conv3_1
    *layer[10] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 10, 
        batchSize, 128, 56, 56, 
        1, 1, 1, 1, 
        256, 3, 3
    );
    layer[11] = (Layer*) malloc(sizeof(Layer));
    *layer[11] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 11, 
        batchSize, 256, 56, 56,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[12] = (Layer*) malloc(sizeof(Layer)); // conv3_2
    *layer[12] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 12, 
        batchSize, 256, 56, 56, 
        1, 1, 1, 1, 
        256, 3, 3
    );
    layer[13] = (Layer*) malloc(sizeof(Layer));
    *layer[13] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 13, 
        batchSize, 256, 56, 56,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[14] = (Layer*) malloc(sizeof(Layer)); // conv3_3
    *layer[14] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 14, 
        batchSize, 256, 56, 56, 
        1, 1, 1, 1, 
        256, 3, 3
    );
    layer[15] = (Layer*) malloc(sizeof(Layer));
    *layer[15] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 15, 
        batchSize, 256, 56, 56,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[16] = (Layer*) malloc(sizeof(Layer)); // pooling
    *layer[16] = Layer(
        vdnnType, algoMode, POOL, 
        &cudnnHandle, &stream_compute, &stream_memory, 16, 
        batchSize, 256, 56, 56, 
        0, 0, 2, 2, 
        0, 2, 2
    );
    layer[17] = (Layer*) malloc(sizeof(Layer)); // conv4_1
    *layer[17] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 17, 
        batchSize, 256, 28, 28, 
        1, 1, 1, 1, 
        512, 3, 3
    );
    layer[18] = (Layer*) malloc(sizeof(Layer));
    *layer[18] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 18, 
        batchSize, 512, 28, 28,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[19] = (Layer*) malloc(sizeof(Layer)); // conv4_2
    *layer[19] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 19, 
        batchSize, 512, 28, 28, 
        1, 1, 1, 1, 
        512, 3, 3
    );
    layer[20] = (Layer*) malloc(sizeof(Layer));
    *layer[20] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 20, 
        batchSize, 512, 28, 28,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[21] = (Layer*) malloc(sizeof(Layer)); // conv4_3
    *layer[21] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 21, 
        batchSize, 512, 28, 28, 
        1, 1, 1, 1, 
        512, 3, 3
    );
    layer[22] = (Layer*) malloc(sizeof(Layer));
    *layer[22] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 22, 
        batchSize, 512, 28, 28,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[23] = (Layer*) malloc(sizeof(Layer)); // pooling
    *layer[23] = Layer(
        vdnnType, algoMode, POOL, 
        &cudnnHandle, &stream_compute, &stream_memory, 23, 
        batchSize, 512, 28, 28, 
        0, 0, 2, 2, 
        0, 2, 2
    );
    layer[24] = (Layer*) malloc(sizeof(Layer)); // conv5_1
    *layer[24] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 24, 
        batchSize, 512, 14, 14, 
        1, 1, 1, 1, 
        512, 3, 3
    );
    layer[25] = (Layer*) malloc(sizeof(Layer));
    *layer[25] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 25, 
        batchSize, 512, 14, 14,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[26] = (Layer*) malloc(sizeof(Layer)); // conv5_2
    *layer[26] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 26, 
        batchSize, 512, 14, 14, 
        1, 1, 1, 1, 
        512, 3, 3
    );
    layer[27] = (Layer*) malloc(sizeof(Layer));
    *layer[27] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 27, 
        batchSize, 512, 14, 14,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[28] = (Layer*) malloc(sizeof(Layer)); // conv5_3
    *layer[28] = Layer(
        vdnnType, algoMode, FC_CONV, 
        &cudnnHandle, &stream_compute, &stream_memory, 28, 
        batchSize, 512, 14, 14, 
        1, 1, 1, 1, 
        512, 3, 3
    );
    layer[29] = (Layer*) malloc(sizeof(Layer));
    *layer[29] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 29, 
        batchSize, 512, 14, 14,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[30] = (Layer*) malloc(sizeof(Layer)); // pooling
    *layer[30] = Layer(
        vdnnType, algoMode, POOL, 
        &cudnnHandle, &stream_compute, &stream_memory, 30, 
        batchSize, 512, 14, 14, 
        0, 0, 2, 2, 
        0, 2, 2
    );
    layer[31] = (Layer*) malloc(sizeof(Layer)); // FC_GEMM1
    *layer[31] = Layer(
        vdnnType, algoMode, FC_GEMM, 
        &cublasHandle, &stream_compute, &stream_memory, 31, 
        batchSize, 512 * 7 * 7, 1, 1,
        0, 0, 0, 0,
        4096, 1, 1
    );
    layer[32] = (Layer*) malloc(sizeof(Layer));
    *layer[32] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 32, 
        batchSize, 4096, 1, 1,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[33] = (Layer*) malloc(sizeof(Layer)); // FC_GEMM2
    *layer[33] = Layer(
        vdnnType, algoMode, FC_GEMM, 
        &cublasHandle, &stream_compute, &stream_memory, 33, 
        batchSize, 4096, 1, 1,
        0, 0, 0, 0,
        4096, 1, 1
    );
    layer[34] = (Layer*) malloc(sizeof(Layer));
    *layer[34] = Layer(
        vdnnType, algoMode, RELU, 
        &cudnnHandle, &stream_compute, &stream_memory, 34, 
        batchSize, 4096, 1, 1,
        0, 0, 0, 0,
        0, 0, 0
    );
    layer[35] = (Layer*) malloc(sizeof(Layer)); // FC_GEMM3
    *layer[35] = Layer(
        vdnnType, algoMode, FC_GEMM, 
        &cublasHandle, &stream_compute, &stream_memory, 35, 
        batchSize, 4096, 1, 1,
        0, 0, 0, 0,
        1000, 1, 1
    );
    layer[36] = (Layer*) malloc(sizeof(Layer));
    *layer[36] = Layer(
        vdnnType, algoMode, SOFTMAX, 
        &cudnnHandle, &stream_compute, &stream_memory, 36, 
        batchSize, 1000, 1, 1,
        0, 0, 0, 0,
        0, 0, 0
    );
}

void Network::setLayersCustomCNN() {
    // CUSTOM CNN
}

void Network::randomInitializeData(void* data, int size) {
    /* Initialize device data, assuming memory is already allocated */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<DATATYPE> dis(-RANDOM_VALUE_LIMIT, RANDOM_VALUE_LIMIT);

    DATATYPE* data_h = new DATATYPE[size];
    for (int i = 0; i < size; i++) data_h[i] = dis(gen);
    checkCuda(cudaMemcpy(data, data_h, sizeof(DATATYPE) * size, cudaMemcpyHostToDevice));
    delete data_h;
}

void Network::sanityCheck() {
    /* Check there are 'numLayers' number of layers */
    for (int i = 0; i < DEFAULT_NUM_LAYER; i++) {
        if (i < numLayers) { M_ASSERT(layer[i] != NULL, "Not enough layers!"); }
        else { M_ASSERT(layer[i] == NULL, "Too many layers!"); }
    }

    /* Check if input of layer 0 has same dimensions as input of network */
    M_ASSERT(
        (layer[0]->InputN() == n_in && 
         layer[0]->InputC() == c_in && 
         layer[0]->InputH() == h_in && 
         layer[0]->InputW() == w_in),
        "Layer 0 input and network input have different dimensions"
    );
    
    /* Check if input of layer L has same dimensions as label of network */
    M_ASSERT(
        (layer[numLayers-1]->OutputN() == n_out && 
         layer[numLayers-1]->OutputC() == c_out && 
         layer[numLayers-1]->OutputH() == h_out && 
         layer[numLayers-1]->OutputW() == w_out),
        "Layer L output and network output have different dimensions"
    );

    /* Check if output of layer l has same dimensions as input of layer l+1 */
    int check;
    for (int i = 0; i < numLayers - 1; i++) {
        if (layer[i+1]->LayerType() == FC_GEMM) {
            check = (layer[i]->DstDataSize() == layer[i+1]->SrcDataSize());
        }
        else {
            check = (layer[i]->OutputN() == layer[i+1]->InputN() && 
                     layer[i]->OutputC() == layer[i+1]->InputC() && 
                     layer[i]->OutputH() == layer[i+1]->InputH() && 
                     layer[i]->OutputW() == layer[i+1]->InputW());
        }
        if (!check) {
            printf("Sanity check error at layer %d and %d!\n", i, i + 1);
            M_ASSERT(0, "Sanity check error!");
        }
    }
}

void Network::setVdnnType(vdnn_t newVdnnType) {
    vdnnType = newVdnnType;

    for (int i = 0; i < DEFAULT_NUM_LAYER; i++) {
        /* Set offload info to false */
        isOffload[i] = false;
        isPrefetched[i] = false;

        /* Free all host-side offload memory space */
        if (offloadedSrcData_h[i] != NULL)
            cudaFreeHost(offloadedSrcData_h[i]);
        offloadedSrcData_h[i] = NULL;
        
        /* Set vdnnMode of all layers to newVdnnType */
        if (layer[i] != NULL)
            layer[i]->setVdnnMode(newVdnnType);
    }

    switch(newVdnnType) {
        case BASELINE:
        {
            /* Initialize dx, dy */
            int dxdySize = findMaxDxDySize();
            ASSERT_EQ(cnmemMalloc(&networkDiffData, dxdySize, NULL));
            ASSERT_EQ(cnmemMalloc(&networkGradData, dxdySize, NULL));

            /* Initialize workspace */
            int workSpaceSize = findMaxWorkSpaceSize();
            if (workSpaceSize > 0)
                ASSERT_EQ(cnmemMalloc(&workSpace, workSpaceSize, NULL));
        }
        break;

        case VDNN_NONE:
        {
        }
        break;

        case VDNN_ALL:
        {
            /* Initialize offload settings and pinned data */
            for (int l = 0; l < numLayers - 1; l++) {
                isOffload[l] = true;
                cudaMallocHost(
                    (void**) &offloadedSrcData_h[l], 
                    layer[l]->SrcDataSize() * sizeof(DATATYPE)
                );
            }
        }
        break;

        case VDNN_CONV:
        {
            for (int l = 0; l < numLayers - 1; l++) {
                if (layer[l]->LayerType() == FC_CONV || layer[l]->LayerType() == CONV) {
                    isOffload[l] = true;
                    cudaMallocHost(
                        (void**) &offloadedSrcData_h[l], 
                        layer[l]->SrcDataSize() * sizeof(DATATYPE)
                    );
                }
            }
        }
        break;
    }
}

void Network::setAlgoType(vdnnAlgoMode_t newAlgoMode) {

}

void Network::vdnnDynamicAlgo() {
    int ret = 0;

    /* 1. VDNN_ALL and VDNN_MEMORY_OPT_ALGO: check feasibility */
    printf("\n1. Test VDNN_ALL and VDNN_MEMORY_OPT_ALGO\n");
    if (forwardPropagation(true) < 0) {
        printf("This CNN model cannot be trained\n");
        assert(0);
    }
    if (backwardPropagation(true) < 0) {
        printf("This CNN model cannot be trained\n");
        assert(0);
    }

    /* 2. VDNN_NONE and VDNN_PERF_OPT_ALGO */
    printf("\n2. Test VDNN_NONE and VDNN_PERF_OPT_ALGO\n");
    reset(cnnType, VDNN_NONE, VDNN_PERF_OPT_ALGO);
    ret = forwardPropagation(true);
    if (ret >= 0) {
        ret = backwardPropagation(true);
        if (ret >= 0) {
            reset(cnnType, VDNN_NONE, VDNN_PERF_OPT_ALGO);
            printf("VDNN_NONE and VDNN_PERF_OPT_ALGO Passed!\n");
            return;
        }
    }
    
    /* 3. VDNN_CONV and VDNN_PERF_OPT_ALGO */
    printf("\n3. Test VDNN_CONV and VDNN_PERF_OPT_ALGO\n");
    reset(cnnType, VDNN_CONV, VDNN_PERF_OPT_ALGO);
    ret = forwardPropagation(true);
    if (ret >= 0) {
        ret = backwardPropagation(true);
        if (ret >= 0) {
            reset(cnnType, VDNN_CONV, VDNN_PERF_OPT_ALGO);
            printf("VDNN_CONV and VDNN_PERF_OPT_ALGO Passed!\n");
            return;
        }
    }

    /* 4. VDNN_ALL and VDNN_PERF_OPT_ALGO */
    printf("\n4. VDNN_ALL and VDNN_PERF_OPT_ALGO\n");
    reset(cnnType, VDNN_ALL, VDNN_PERF_OPT_ALGO);
    ret = forwardPropagation(true);
    if (ret >= 0) {
        ret = backwardPropagation(true);
        if (ret >= 0) {
            reset(cnnType, VDNN_ALL, VDNN_PERF_OPT_ALGO);
            printf("VDNN_ALL and VDNN_PERF_OPT_ALGO Passed!\n");
            return;
        }
    }

    /* 5. DYNAMIC_ALGO */
    printf("\n5. Dynamic Algorithm with VDNN_ALL\n");
    reset(cnnType, VDNN_ALL, VDNN_PERF_OPT_ALGO);
    for (int id = 0; id < DEFAULT_NUM_LAYER; id++) {
        if (layer[id] == NULL)
            continue;
        if (layer[id]->LayerType() == FC_CONV || layer[id]->LayerType() == CONV)
            layer[id]->findDynamicAlgo();
    }
}

int Network::findMaxDxDySize() {
    int ret = 0;
    int temp;
    for (int i = 0; i < numLayers; i++) {
        temp = layer[i]->SrcDataSize();
        if (ret < temp)
            ret = temp;
    }
    temp = layer[numLayers-1]->DstDataSize();
    if (ret < temp)
        ret = temp;
    /* Return value is in bytes, not DATATYPE */
    return ret * sizeof(DATATYPE);
}

int Network::findMaxWorkSpaceSize() {
    int ret = 0;
    int temp;
    for (int i = 0; i < numLayers; i++) {
        if (layer[i]->LayerType() == FC_CONV || layer[i]->LayerType() == CONV) {
            temp = layer[i]->FwdWorkSpaceSize();
            if (ret < temp) ret = temp;
            temp = layer[i]->BwdDataWorkSpaceSize();
            if (ret < temp) ret = temp;
            temp = layer[i]->BwdFilterWorkSpaceSize();
            if (ret < temp) ret = temp;
        }
    }
    /* Return value is in bytes, not DATATYPE */
    return ret;
}

void Network::calculateFirstDiffData(void* result, void* first, void* second) {
    /* result = first - second */
    int labelDataSize = getOutputSize();
    randomInitializeData(labelData, labelDataSize);
    int gridSize = CEIL(labelDataSize, BLOCK_SIZE);
    int blockSize = BLOCK_SIZE;
    calculateDiffTensor<<<gridSize, blockSize>>>(
        (DATATYPE*) result, 
        (DATATYPE*) first, 
        (DATATYPE*) second, 
        labelDataSize
    );
}

int Network::forwardPropagation(bool isDynamic) {
    size_t free, total;
    if (vdnnType == BASELINE) {
        randomInitializeData(inputData, getInputSize());
        for (int l = 0; l < numLayers; l++) {
            if (l == 0)
                layer[l]->copySrcData(inputData);
            else
                layer[l]->copySrcData(layer[l-1]->DstData());
            if (layer[l]->forward(isOffload, offloadedSrcData_h, workSpace, isDynamic) < 0)
                return -1;

            /* Track memory usage */
            cnmemMemGetInfo(&free, &total, NULL);
            printf(
                "[Fwd] Layer %d, Free Memory: %lu, Total Memory: %lu, Memory Usage: %lu\n", 
                l, free, total, 
                total - free + layer[l]->FwdWorkSpaceSize()
            );
        }
    }
    else if (vdnnType == VDNN_NONE || vdnnType == VDNN_ALL || vdnnType == VDNN_CONV) {
        randomInitializeData(inputData, getInputSize());
        for (int l = 0; l < numLayers; l++) {
            if (l == 0)
                layer[l]->copySrcData(inputData);
            else
                layer[l]->copySrcData(layer[l-1]->DstData());

            /* Forward propagation */
            if (layer[l]->forward(isOffload, offloadedSrcData_h, workSpace, isDynamic) < 0)
                return -1;

            /* Synchronize stream_compute and stream_memory */
            checkCuda(cudaStreamSynchronize(stream_compute));
            checkCuda(cudaStreamSynchronize(stream_memory));

            /* Release srcData of this layer */
            if (l != 0 && isOffload[l]) 
                ASSERT_EQ(cnmemFree(layer[l]->SrcData(), NULL));

            /* Track memory usage */
            cnmemMemGetInfo(&free, &total, NULL);
            printf(
                "[Fwd] Layer %d, Free Memory: %lu, Total Memory: %lu, Memory Usage: %lu\n", 
                l, free, total, 
                total - free + layer[l]->FwdWorkSpaceSize()
            );
        }
    }
    else {
        M_ASSERT(0, "Should not reach here");
    }
    return 0;
}

int Network::backwardPropagation(bool isDynamic) {
    size_t free, total;
    if (vdnnType == BASELINE) {
        calculateFirstDiffData(networkDiffData, layer[numLayers-1]->DstData(), labelData);
        for (int l = numLayers-1; l >= 0; l--) {
            if (l == numLayers-1) {
                layer[l]->copyDiffData(networkDiffData);
                layer[l]->copyGradData(networkGradData);
            }
            else {
                layer[l]->copyDiffData(layer[l+1]->GradData());
                layer[l]->copyGradData(layer[l+1]->DiffData());
            }
            if (layer[l]->backward(-1, NULL, NULL, 0, workSpace, isDynamic) < 0)
                return -1;

            /* Track memory usage */
            cnmemMemGetInfo(&free, &total, NULL);
            printf(
                "[Bwd] Layer %d, Free Memory: %lu, Total Memory: %lu, Memory Usage: %lu\n", 
                l, free, total, 
                total - free + layer[l]->BwdDataWorkSpaceSize() + layer[l]->BwdFilterWorkSpaceSize()
            );
        }
    }
    else if (vdnnType == VDNN_NONE || vdnnType == VDNN_ALL || vdnnType == VDNN_CONV) {
        resetIsPrefetchedArray();
        if (layer[numLayers-1]->cnmemMallocDiffData(isDynamic) < 0)
            return -1;
        calculateFirstDiffData(layer[numLayers-1]->DiffData(), layer[numLayers-1]->DstData(), labelData);

        for (int l = numLayers-1; l >= 0; l--) {
            if (l != numLayers-1) {
                layer[l]->copyDiffData(layer[l+1]->GradData());
                layer[l]->copyDstData(layer[l+1]->SrcData());
            }

            if (l == 0) {
                if (layer[l]->backward(-1, NULL, NULL, 0, workSpace, isDynamic) < 0)
                    return -1;
            }
            else {
                int prefId = findPrefetchLayer(l);
                if (prefId >= 0) {
                    if (layer[prefId]->cnmemMallocSrcData(isDynamic) < 0)
                        return -1;
                    if (layer[l]->backward(prefId, offloadedSrcData_h[prefId], layer[prefId]->SrcData(), sizeof(DATATYPE) * layer[prefId]->SrcDataSize(), workSpace, isDynamic) < 0)
                        return -1;
                }
                else {
                    if (layer[l]->backward(-1, NULL, NULL, 0, workSpace, isDynamic) < 0)
                        return -1;
                }
            }

            /* Synchronize stream_compute and stream_memory */
            checkCuda(cudaStreamSynchronize(stream_compute));
            checkCuda(cudaStreamSynchronize(stream_memory));

            /* Release dstData and diffData */
            ASSERT_EQ(cnmemFree(layer[l]->DstData(), NULL));
            ASSERT_EQ(cnmemFree(layer[l]->DiffData(), NULL));

            /* Track memory usage */
            cnmemMemGetInfo(&free, &total, NULL);
            printf(
                "[Bwd] Layer %d, Free Memory: %lu, Total Memory: %lu, Memory Usage: %lu\n", 
                l, free, total, 
                total - free + layer[l]->BwdDataWorkSpaceSize() + layer[l]->BwdFilterWorkSpaceSize()
            );
        }
        
        ASSERT_EQ(cnmemFree(layer[0]->GradData(), NULL));
        inputData = layer[0]->SrcData();
    }
    else {
        M_ASSERT(0, "Should not reach here");
    }
    return 0;
}

void Network::resetIsPrefetchedArray() {
    for (int i = 0; i < DEFAULT_NUM_LAYER; i++)
        isPrefetched[i] = false;
}

int Network::findPrefetchLayer(int currLayerId) {
    for (int id = currLayerId-1; id >= 0; id--) {
        if (isOffload[id] == true && isPrefetched[id] == false) {
            isPrefetched[id] = true;
            return id;
        }
        else if (layer[id]->LayerType() == FC_CONV || layer[id]->LayerType() == CONV)
            return -1;
    }
    return -1;
}