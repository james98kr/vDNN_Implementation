#include "layer.h"

using namespace std;

cudnnConvolutionFwdAlgo_t fwdAlgoList[NUM_FWD_ALGO] = {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
};

cudnnConvolutionBwdDataAlgo_t bwdDataAlgoList[NUM_BWD_DATA_ALGO] = {
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
};

cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgoList[NUM_BWD_FILTER_ALGO] = {
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING
};

__global__ void fillVectorWithOnes(DATATYPE* vec, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size)
        return;
    vec[idx] = 1.0f;
}

Layer::Layer(
    vdnn_t          _vdnnMode,
    vdnnAlgoMode_t  _vdnnAlgoMode,
	Layer_t		    _layerType,
    cublasHandle_t* _cublasHandle,
	cudaStream_t*	_stream_compute,
	cudaStream_t*	_stream_memory,
	int _id,
	int _n, int _c, int _h, int _w,
	int _pad_h, int _pad_w, int _stride_h, int _stride_w,
	int _k, int _r, int _s):  
    vdnnMode(_vdnnMode), vdnnAlgoMode(_vdnnAlgoMode),
    layerType(_layerType), cublasHandle(_cublasHandle), 
    stream_compute(_stream_compute), stream_memory(_stream_memory), 
    layerId(_id), n_in(_n), c_in(_c), h_in(_h), w_in(_w),
    pad_h(_pad_h), pad_w(_pad_w), stride_h(_stride_h), stride_w(_stride_w),
	k(_k), r(_r), s(_s)
{
    /* Basic settings */
    insideInceptionModule  = false;
    headOfFork             = false;
    tailOfFork             = false;
    forkId                 = -1;
    refCntFwd              = 0;
    refCntBwd              = 0;
    inPlaceOp              = false;
    srcData                = NULL; // x
    filterData             = NULL; // w
    biasData               = NULL; // bias
    dstData                = NULL; // y
    diffData               = NULL; // dy
    gradData               = NULL; // dx
    fwdWorkSpaceSize       = 0;
    bwdDataWorkSpaceSize   = 0;
    bwdFilterWorkSpaceSize = 0;
    producerLayerId.clear();
    consumerLayerId.clear();

    /* Make sure only FC layer is initialized with cuBLAS */
    M_ASSERT(_layerType == FC_GEMM, "Layer type is not FC_GEMM!");
    M_ASSERT(h_in == 1 && w_in == 1 && r == 1 && s == 1, "Invalid dimensions!");

    // Input: (c_in, n_out)
    // Weight: (c_in, c_out), will be transposed
    // Bias: (c_out, n_out)
    // Output: (c_out, n_out)
    n_out = n_in;
    c_out = k;
    h_out = 1;
    w_out = 1;

    initializeFullyConnectedFilterAndBias();

    /* Allocate GPU memory for dstData */
    if (vdnnMode == BASELINE) cnmemMallocDstData(false);

    printf("----------------------- Layer %d -----------------------\n", layerId);
    printf("Layer Type: FC_GEMM\n");
    printf("Input (n, c, h, w): (%d, %d, %d, %d)\n", n_in, c_in, h_in, w_in);
    printf("Output (n, c, h, w): (%d, %d, %d, %d)\n", n_out, c_out, h_out, w_out);
    printf("Filter (k, c, r, s): (%d, %d, %d, %d)\n", k, c_in, r, s);
    printf("Stride: (%d, %d), Padding: (%d, %d)\n", stride_h, stride_w, pad_h, pad_w);
}

Layer::Layer(
    vdnn_t          _vdnnMode,
    vdnnAlgoMode_t  _vdnnAlgoMode,
	Layer_t		    _layerType,
	cudnnHandle_t*	_cudnnHandle,
	cudaStream_t*	_stream_compute,
	cudaStream_t*	_stream_memory,
	int _id,
	int _n, int _c, int _h, int _w,
	int _pad_h, int _pad_w, int _stride_h, int _stride_w,
	int _k, int _r, int _s):  
    vdnnMode(_vdnnMode), vdnnAlgoMode(_vdnnAlgoMode),
    layerType(_layerType), cudnnHandle(_cudnnHandle), 
    stream_compute(_stream_compute), stream_memory(_stream_memory), 
    layerId(_id), n_in(_n), c_in(_c), h_in(_h), w_in(_w),
    pad_h(_pad_h), pad_w(_pad_w), stride_h(_stride_h), stride_w(_stride_w),
	k(_k), r(_r), s(_s)
{
    /* Basic settings */
    insideInceptionModule  = false;
    headOfFork             = false;
    tailOfFork             = false;
    forkId                 = -1;
    refCntFwd              = 0;
    refCntBwd              = 0;
    inPlaceOp              = false;
    srcData                = NULL; // x
    filterData             = NULL; // w
    biasData               = NULL; // bias
    dstData                = NULL; // y
    diffData               = NULL; // dy
    gradData               = NULL; // dx
    fwdWorkSpaceSize       = 0;
    bwdDataWorkSpaceSize   = 0;
    bwdFilterWorkSpaceSize = 0;
    producerLayerId.clear();
    consumerLayerId.clear();
    string type;

    /* Layer initialization for each type */
    switch(_layerType) {

        case FC_GEMM:
        {
            M_ASSERT(false, "FC_GEMM type layer cannot be initialized with cuDNN!");
        }
        break;

        case CONV:
        case FC_CONV:
        {
            type = "FC_CONV";
            /* Input descriptor setup */
            checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
            checkCUDNN(cudnnSetTensor4dDescriptor(
                srcTensorDesc,
                /*format=*/CUDNN_TENSOR_FORMAT,
                /*dataType=*/CUDNN_DATATYPE,
                /*batch_size=*/n_in,
                /*channels=*/c_in,
                /*image_height=*/h_in,
                /*image_width=*/w_in
            ));

            /* dx descriptor setup */
            checkCUDNN(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
            checkCUDNN(cudnnSetTensor4dDescriptor(
                srcDiffTensorDesc,
                /*format=*/CUDNN_TENSOR_FORMAT,
                /*dataType=*/CUDNN_DATATYPE,
                /*batch_size=*/n_in,
                /*channels=*/c_in,
                /*image_height=*/h_in,
                /*image_width=*/w_in
            ));

            /* Filter descriptor setup */
            checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
            checkCUDNN(cudnnSetFilter4dDescriptor(
                filterDesc,
                /*dataType=*/CUDNN_DATATYPE,
                /*format=*/CUDNN_TENSOR_FORMAT,
                /*out_channels=*/k,
                /*in_channels=*/c_in,
                /*kernel_height=*/r,
                /*kernel_width=*/s
            ));

            /* Convolution descriptor setup */
            checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
            checkCUDNN(cudnnSetConvolution2dDescriptor(
                convDesc,
                /*pad_height=*/pad_h,
                /*pad_width=*/pad_w,
                /*vertical_stride=*/stride_h,
                /*horizontal_stride=*/stride_w,
                /*dilation_height=*/1,
                /*dilation_width=*/1,
                /*mode=*/CUDNN_CONV_MODE,
                /*computeType=*/CUDNN_DATATYPE
            ));

            /* Output descriptor setup */
            checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
            checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
                convDesc,
                srcTensorDesc,
                filterDesc,
                &n_out,
                &c_out,
                &h_out,
                &w_out
            ));
            checkCUDNN(cudnnSetTensor4dDescriptor(
                dstTensorDesc,
                /*format=*/CUDNN_TENSOR_FORMAT,
                /*dataType=*/CUDNN_DATATYPE,
                /*batch_size=*/n_out,
                /*channels=*/c_out,
                /*image_height=*/h_out,
                /*image_width=*/w_out
            ));

            /* dy descriptor setup */
            checkCUDNN(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
            checkCUDNN(cudnnSetTensor4dDescriptor(
                dstDiffTensorDesc,
                /*format=*/CUDNN_TENSOR_FORMAT,
                /*dataType=*/CUDNN_DATATYPE,
                /*batch_size=*/n_out,
                /*channels=*/c_out,
                /*image_height=*/h_out,
                /*image_width=*/w_out
            ));

            /* Bias descriptor setup */
            checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
            checkCUDNN(cudnnSetTensor4dDescriptor(
                biasTensorDesc,
                /*format=*/CUDNN_TENSOR_FORMAT,
                /*dataType=*/CUDNN_DATATYPE,
                /*batch_size=*/1,
                /*channels=*/c_out,
                /*image_height=*/1,
                /*image_width=*/1
            ));

            /* Random initialization of filter and bias data using
               randomly generated DATATYPE number between -1 and 1 */
            initializeConvolutionFilterAndBias();

            /* Find appropriate convolution algorithm */
            if (_vdnnAlgoMode == VDNN_MEMORY_OPT_ALGO) {
                findMemOptFwdAlgo();
                findMemOptBwdFilterAlgo();
                findMemOptBwdDataAlgo();
            }
            else if (_vdnnAlgoMode == VDNN_PERF_OPT_ALGO) {
                findPerfOptFwdAlgo();
                findPerfOptBwdFilterAlgo();
                findPerfOptBwdDataAlgo();
            }

            /* Find workspace size for fwd/bwd propagation */
            checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                *cudnnHandle,
                srcTensorDesc,
                filterDesc,
                convDesc,
                dstTensorDesc,
                fwdAlgo,
                &fwdWorkSpaceSize
            ));
            checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
                *cudnnHandle,
                filterDesc,
                dstDiffTensorDesc,
                convDesc,
                srcDiffTensorDesc,
                bwdDataAlgo,
                &bwdDataWorkSpaceSize
            ));
            checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                *cudnnHandle,
                srcTensorDesc,
                dstDiffTensorDesc,
                convDesc,
                filterDesc,
                bwdFilterAlgo,
                &bwdFilterWorkSpaceSize
            ));
        }
        break;

        case RELU:
        case TANH:
        case SIGMOID:
        case SOFTMAX:
        {
            n_out = n_in;
            c_out = c_in;
            h_out = h_in;
            w_out = w_in;

            /* Input descriptor setup */
            checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
            checkCUDNN(cudnnSetTensor4dDescriptor(
                srcTensorDesc,
                /*format=*/CUDNN_TENSOR_FORMAT,
                /*dataType=*/CUDNN_DATATYPE,
                /*batch_size=*/n_in,
                /*channels=*/c_in,
                /*image_height=*/h_in,
                /*image_width=*/w_in
            ));

            /* dx descriptor setup */
            checkCUDNN(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
            checkCUDNN(cudnnSetTensor4dDescriptor(
                srcDiffTensorDesc,
                /*format=*/CUDNN_TENSOR_FORMAT,
                /*dataType=*/CUDNN_DATATYPE,
                /*batch_size=*/n_in,
                /*channels=*/c_in,
                /*image_height=*/h_in,
                /*image_width=*/w_in
            ));

            /* Output descriptor setup */
            checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
            checkCUDNN(cudnnSetTensor4dDescriptor(
                dstTensorDesc,
                /*format=*/CUDNN_TENSOR_FORMAT,
                /*dataType=*/CUDNN_DATATYPE,
                /*batch_size=*/n_out,
                /*channels=*/c_out,
                /*image_height=*/h_out,
                /*image_width=*/w_out
            ));

            /* dy descriptor setup */
            checkCUDNN(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
            checkCUDNN(cudnnSetTensor4dDescriptor(
                dstDiffTensorDesc,
                /*format=*/CUDNN_TENSOR_FORMAT,
                /*dataType=*/CUDNN_DATATYPE,
                /*batch_size=*/n_out,
                /*channels=*/c_out,
                /*image_height=*/h_out,
                /*image_width=*/w_out
            ));

            /* Selection of activation function */
            cudnnActivationMode_t activationMode;
            if (_layerType == RELU) {
                type = "RELU";
                activationMode = CUDNN_ACTIVATION_RELU;
            }
            else if (_layerType == TANH) {
                type = "TANH";
                activationMode = CUDNN_ACTIVATION_TANH;
            }
            else if (_layerType == SIGMOID) {
                type = "SIGMOID";
                activationMode = CUDNN_ACTIVATION_SIGMOID;
            }
            else if (_layerType == SOFTMAX) {
                type = "SOFTMAX";
                break;
            }
            else {
                M_ASSERT(false, "Invalid activation function type!");
            }
                
            /* Activation descriptor setup */
            checkCUDNN(cudnnCreateActivationDescriptor(&actvDesc));
            checkCUDNN(cudnnSetActivationDescriptor(
                actvDesc, 
                activationMode,
                CUDNN_PROPAGATE_NAN,
                0.0
            ));
        }
        break;

        case POOL:
        {
            type = "POOL";
            /* Input descriptor setup */
            checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
            checkCUDNN(cudnnSetTensor4dDescriptor(
                srcTensorDesc,
                /*format=*/CUDNN_TENSOR_FORMAT,
                /*dataType=*/CUDNN_DATATYPE,
                /*batch_size=*/n_in,
                /*channels=*/c_in,
                /*image_height=*/h_in,
                /*image_width=*/w_in
            ));

            /* dx descriptor setup */
            checkCUDNN(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
            checkCUDNN(cudnnSetTensor4dDescriptor(
                srcDiffTensorDesc,
                /*format=*/CUDNN_TENSOR_FORMAT,
                /*dataType=*/CUDNN_DATATYPE,
                /*batch_size=*/n_in,
                /*channels=*/c_in,
                /*image_height=*/h_in,
                /*image_width=*/w_in
            ));

            /* Pooling descriptor setup */
            checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
            checkCUDNN(cudnnSetPooling2dDescriptor(
                poolingDesc, 
                CUDNN_POOLING_MAX, 
                CUDNN_NOT_PROPAGATE_NAN, 
                /*windowHeight=*/r, 
                /*windowWidth=*/s, 
                /*verticalPadding=*/pad_h, 
                /*horizontalPadding=*/pad_w, 
                /*verticalStride=*/stride_h, 
                /*horizontalStride=*/stride_w
            ));

            /* Output descriptor setup */
            checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
            checkCUDNN(cudnnGetPooling2dForwardOutputDim(
                poolingDesc,
                srcTensorDesc,
                &n_out,
                &c_out,
                &h_out,
                &w_out
            ));
            checkCUDNN(cudnnSetTensor4dDescriptor(
                dstTensorDesc,
                /*format=*/CUDNN_TENSOR_FORMAT,
                /*dataType=*/CUDNN_DATATYPE,
                /*batch_size=*/n_out,
                /*channels=*/c_out,
                /*image_height=*/h_out,
                /*image_width=*/w_out
            ));

            /* dy descriptor setup */
            checkCUDNN(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
            checkCUDNN(cudnnSetTensor4dDescriptor(
                dstDiffTensorDesc,
                /*format=*/CUDNN_TENSOR_FORMAT,
                /*dataType=*/CUDNN_DATATYPE,
                /*batch_size=*/n_out,
                /*channels=*/c_out,
                /*image_height=*/h_out,
                /*image_width=*/w_out
            ));
        }
        break;

        case CONCATENATE:
        {

        }
        break;
    }

    /* Allocate GPU memory for dstData */
    if (vdnnMode == BASELINE) cnmemMallocDstData(false);

    printf("----------------------- Layer %d -----------------------\n", layerId);
    cout << "Layer Type: " << type << endl;
    printf("Input (n, c, h, w): (%d, %d, %d, %d)\n", n_in, c_in, h_in, w_in);
    printf("Output (n, c, h, w): (%d, %d, %d, %d)\n", n_out, c_out, h_out, w_out);
    printf("Filter (k, c, r, s): (%d, %d, %d, %d)\n", k, c_in, r, s);
    printf("Stride: (%d, %d), Padding: (%d, %d)\n", stride_h, stride_w, pad_h, pad_w);
    if (layerType == FC_CONV || layerType == CONV) {
        printf("CONV Layer %d fwdWorkSpaceSize: %lu\n", layerId, fwdWorkSpaceSize);
        printf("CONV Layer %d bwdDataWorkSpaceSize: %lu\n", layerId, bwdDataWorkSpaceSize);
        printf("CONV Layer %d bwdFilterWorkSpaceSize: %lu\n", layerId, bwdFilterWorkSpaceSize);
    }
}

Layer::~Layer() {}

int Layer::forward(bool* _offloaded, DATATYPE** _offloadedSrcData_h, void* workSpace, bool isDynamic) {
    int ret = 0;
    switch(layerType) {
        case FC_GEMM:
            ret = fullyConnectedForward(_offloaded, _offloadedSrcData_h, workSpace, isDynamic);
            break;
        case CONV:
        case FC_CONV:
            ret = convolutionForward(_offloaded, _offloadedSrcData_h, workSpace, isDynamic);
            break;
        case RELU:
        case TANH:
        case SIGMOID:
            ret = activationForward(_offloaded, _offloadedSrcData_h, workSpace, isDynamic);
            break;
        case SOFTMAX:
            ret = softmaxForward(_offloaded, _offloadedSrcData_h, workSpace, isDynamic);
            break;
        case POOL:
            ret = poolingForward(_offloaded, _offloadedSrcData_h, workSpace, isDynamic);
            break;
    }
    return ret;
}

int Layer::backward(int _layer_id_to_prefetch, DATATYPE* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes, void* workSpace, bool isDynamic) {
    int ret = 0;
    switch(layerType) {
        case FC_GEMM:
            ret = fullyConnectedBackward(_layer_id_to_prefetch, _prefetchedSrcData_h, _srcDataToPrefetch, _prefetch_bytes, workSpace, isDynamic);
            break;
        case CONV:
        case FC_CONV:
            ret = convolutionBackward(_layer_id_to_prefetch, _prefetchedSrcData_h, _srcDataToPrefetch, _prefetch_bytes, workSpace, isDynamic);
            break;
        case RELU:
        case TANH:
        case SIGMOID:
            ret = activationBackward(_layer_id_to_prefetch, _prefetchedSrcData_h, _srcDataToPrefetch, _prefetch_bytes, workSpace, isDynamic);
            break;
        case SOFTMAX:
            ret = softmaxBackward(_layer_id_to_prefetch, _prefetchedSrcData_h, _srcDataToPrefetch, _prefetch_bytes, workSpace, isDynamic);
            break;
        case POOL:
            ret = poolingBackward(_layer_id_to_prefetch, _prefetchedSrcData_h, _srcDataToPrefetch, _prefetch_bytes, workSpace, isDynamic);
            break;
    }
    return ret;
}

int Layer::convolutionForward(bool* _offloaded, DATATYPE** _offloadedSrcData_h, void* workSpace, bool isDynamic) {
    void* curWorkSpace= NULL;
    const DATATYPE alpha = 1.0f, beta = 0.0f;

    /* Allocate GPU memory for workspace */
    if (workSpace != NULL) {
        M_ASSERT(vdnnMode == BASELINE, "vdnn mode should be BASELINE");
        curWorkSpace = workSpace;
    }
    else if (workSpace == NULL && fwdWorkSpaceSize > 0) {
        if (ASSERT_EQ(cnmemMalloc(&curWorkSpace, fwdWorkSpaceSize, NULL), isDynamic) < 0)
            return -1;
    }
    
    /* Allocate GPU memory for dstData */
    if (vdnnMode != BASELINE) {
        if (cnmemMallocDstData(isDynamic) < 0)
            return -1;
    }

    /* Forward convolution */
    checkCUDNN(cudnnConvolutionForward(   
        *cudnnHandle,
        &alpha,
        srcTensorDesc, srcData,
        filterDesc, filterData,
        convDesc, fwdAlgo,
        curWorkSpace, fwdWorkSpaceSize,
        &beta,
        dstTensorDesc, dstData
    ));

    /* Offload srcData to host */
    if (_offloaded[layerId])
        offloadSrcData(_offloaded, _offloadedSrcData_h);

    /* Add bias to output of forward convolution */
    checkCUDNN(cudnnAddTensor( 
        *cudnnHandle, 
        &alpha, 
        biasTensorDesc, biasData,
        &alpha,
        dstTensorDesc, dstData
    ));

    /* Free workspace memory */
    if (workSpace == NULL && fwdWorkSpaceSize > 0)
        ASSERT_EQ(cnmemFree(curWorkSpace, NULL), isDynamic);
    
    return 0;
}

int Layer::convolutionBackward(int _layer_id_to_prefetch, DATATYPE* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes, void* workSpace, bool isDynamic) {
    void* curWorkSpace = NULL;
    const DATATYPE alpha = 1.0f, beta = 0.0f, lr = LEARNING_RATE * -1.0f;

    /* Allocate GPU memory for gradData */
    if (vdnnMode != BASELINE) {
        if (cnmemMallocGradData(isDynamic) < 0)
            return -1;
    }

    /* Prefetch srcData of previous layer */
    if (vdnnMode != BASELINE && _layer_id_to_prefetch >= 0)
        prefetchPreviousSrcData(_prefetchedSrcData_h, _srcDataToPrefetch, _prefetch_bytes);

    /* Allocate GPU memory for bwd data propagation workspace */
    if (workSpace != NULL) {
        M_ASSERT(vdnnMode == BASELINE, "vdnn mode should be BASELINE");
        curWorkSpace = workSpace;
    }
    else if (workSpace == NULL && bwdDataWorkSpaceSize > 0) {
        if (ASSERT_EQ(cnmemMalloc(&curWorkSpace, bwdDataWorkSpaceSize, NULL), isDynamic) < 0)
            return -1;    
    }
        
    /* Backward Data Propagation */
    checkCUDNN(cudnnConvolutionBackwardData(
        *cudnnHandle,
        &alpha,
        filterDesc, filterData,
        dstDiffTensorDesc, diffData,
        convDesc, bwdDataAlgo,
        curWorkSpace, bwdDataWorkSpaceSize,
        &beta,
        srcDiffTensorDesc, gradData
    ));
    if (workSpace == NULL && bwdDataWorkSpaceSize > 0)
        ASSERT_EQ(cnmemFree(curWorkSpace, NULL));

    /* Backward Bias Propagation: Notice that the 
       learning rate value is used for gradient update */
    checkCUDNN(cudnnConvolutionBackwardBias(
        *cudnnHandle,
        &lr,
        dstDiffTensorDesc, diffData,
        &alpha,
        biasTensorDesc, biasData
    ));

    /* Allocate GPU memory for bwd filter propagation workspace */
    if (workSpace != NULL) {
        M_ASSERT(vdnnMode == BASELINE, "vdnn mode should be BASELINE");
        curWorkSpace = workSpace;
    }
    else if (workSpace == NULL && bwdFilterWorkSpaceSize > 0) {
        if (ASSERT_EQ(cnmemMalloc(&curWorkSpace, bwdFilterWorkSpaceSize, NULL), isDynamic) < 0)
            return -1;
    }

    /* Backward Filter Propagation: Notice that the 
       learning rate value is used for gradient update */
    checkCUDNN(cudnnConvolutionBackwardFilter(
        *cudnnHandle,
        &lr,
        srcTensorDesc, srcData,
        dstDiffTensorDesc, diffData,
        convDesc, bwdFilterAlgo,
        curWorkSpace, bwdFilterWorkSpaceSize,
        &alpha,
        filterDesc, filterData
    ));
    if (workSpace == NULL && bwdFilterWorkSpaceSize > 0)
        ASSERT_EQ(cnmemFree(curWorkSpace, NULL));

    return 0;
}

int Layer::fullyConnectedForward(bool* _offloaded, DATATYPE** _offloadedSrcData_h, void* workSpace, bool isDynamic) {
    const DATATYPE alpha = 1.0f, beta = 0.0f;    
    
    /* Allocate GPU memory for dstData */
    if (vdnnMode != BASELINE) {
        if (cnmemMallocDstData(isDynamic) < 0)
            return -1;
    }

    /* Calculate dstData */
    checkCUBLAS(cublasSgemm(
        *cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
        c_out, n_out, c_in,
        &alpha,
        (DATATYPE*) filterData, c_in,
        (DATATYPE*) srcData, c_in,
        &beta,
        (DATATYPE*) dstData, c_out
    ));

    /* Offload srcData to host */
    if (_offloaded[layerId])
        offloadSrcData(_offloaded, _offloadedSrcData_h);

    /* Extend bias vector and add to dstData */
    void* oneVec;
    int gridSize = CEIL(n_out, BLOCK_SIZE);
    int blockSize = BLOCK_SIZE;
    if (ASSERT_EQ(cnmemMalloc((void**) &oneVec, sizeof(DATATYPE) * n_out, NULL), isDynamic) < 0) return -1;
    fillVectorWithOnes<<<gridSize, blockSize>>>((DATATYPE*) oneVec, n_out);
    checkCUBLAS(cublasSgemm(
        *cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
        c_out, n_out, 1,
        &alpha,
        (DATATYPE*) biasData, c_out,
        (DATATYPE*) oneVec, 1,
        &alpha,
        (DATATYPE*) dstData, c_out
    ));
    ASSERT_EQ(cnmemFree(oneVec, NULL));

    return 0;
}

int Layer::fullyConnectedBackward(int _layer_id_to_prefetch, DATATYPE* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes, void* workSpace, bool isDynamic) {
    const DATATYPE alpha = 1.0f, beta = 0.0f, lr = LEARNING_RATE * -1.0f;

    /* Allocate GPU memory for gradData */
    if (vdnnMode != BASELINE) {
        if (cnmemMallocGradData(isDynamic) < 0)
            return -1;
    }

    /* Prefetch srcData of previous layer */
    if (vdnnMode != BASELINE && _layer_id_to_prefetch >= 0)
        prefetchPreviousSrcData(_prefetchedSrcData_h, _srcDataToPrefetch, _prefetch_bytes);

    /* Calculate gradData = filterData * diffData (W^T * dY) */
    checkCUBLAS(cublasSgemm(
        *cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
        c_in, n_out, c_out,
        &alpha,
        (DATATYPE*) filterData, c_in,
        (DATATYPE*) diffData, c_out,
        &beta,
        (DATATYPE*) gradData, c_in
    ));

    /* Update filterData (dW = srcData * diffData (x * dY)) */
    checkCUBLAS(cublasSgemm(
        *cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
        c_in, c_out, n_out,
        &lr,
        (DATATYPE*) srcData, c_in,
        (DATATYPE*) diffData, c_out,
        &alpha, 
        (DATATYPE*) filterData, c_in
    ));

    /* Update biasData */
    void* oneVec;
    int gridSize = CEIL(n_out, BLOCK_SIZE);
    int blockSize = BLOCK_SIZE;
    if (ASSERT_EQ(cnmemMalloc((void**) &oneVec, sizeof(DATATYPE) * n_out, NULL), isDynamic) < 0) return -1;
    fillVectorWithOnes<<<gridSize, blockSize>>>((DATATYPE*) oneVec, n_out);
    checkCUBLAS(cublasSgemm(
        *cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
        c_out, 1, n_out,
        &lr,
        (DATATYPE*) diffData, c_out,
        (DATATYPE*) oneVec, n_out,
        &alpha,
        (DATATYPE*) biasData, c_out
    ));
    ASSERT_EQ(cnmemFree(oneVec, NULL));

    return 0;
}

int Layer::activationForward(bool* _offloaded, DATATYPE** _offloadedSrcData_h, void* workSpace, bool isDynamic) {
    /* Allocate GPU memory for dstData */
    if (vdnnMode != BASELINE) {
        if (cnmemMallocDstData(isDynamic) < 0)
            return -1;
    }

    const DATATYPE alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnActivationForward(
        *cudnnHandle,
        actvDesc,
        &alpha,
        srcTensorDesc, srcData,
        &beta,
        dstTensorDesc, dstData
    ));

    /* Offload srcData to host */
    if (_offloaded[layerId])
        offloadSrcData(_offloaded, _offloadedSrcData_h);

    return 0;
}

int Layer::activationBackward(int _layer_id_to_prefetch, DATATYPE* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes, void* workSpace, bool isDynamic) {
    /* Allocate GPU memory for gradData */
    if (vdnnMode != BASELINE) {
        if (cnmemMallocGradData(isDynamic) < 0)
            return -1;
    }

    /* Prefetch srcData of previous layer */
    if (vdnnMode != BASELINE && _layer_id_to_prefetch >= 0)
        prefetchPreviousSrcData(_prefetchedSrcData_h, _srcDataToPrefetch, _prefetch_bytes);

    const DATATYPE alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnActivationBackward(
        *cudnnHandle,
        actvDesc,
        &alpha,
        dstTensorDesc, dstData,
        dstDiffTensorDesc, diffData,
        srcTensorDesc, srcData,
        &beta,
        srcDiffTensorDesc, gradData
    ));

    return 0;
}

int Layer::softmaxForward(bool* _offloaded, DATATYPE** _offloadedSrcData_h, void* workSpace, bool isDynamic) {
    /* Allocate GPU memory for dstData */
    if (vdnnMode != BASELINE) {
        if (cnmemMallocDstData(isDynamic) < 0)
            return -1;
    }

    const DATATYPE alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnSoftmaxForward(
        *cudnnHandle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        srcTensorDesc, srcData,
        &beta,
        dstTensorDesc, dstData
    ));

    /* Offload srcData to host */
    if (_offloaded[layerId])
        offloadSrcData(_offloaded, _offloadedSrcData_h);

    return 0;
}

int Layer::softmaxBackward(int _layer_id_to_prefetch, DATATYPE* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes, void* workSpace, bool isDynamic) {
    /* Allocate GPU memory for gradData */
    if (vdnnMode != BASELINE) {
        if (cnmemMallocGradData(isDynamic) < 0)
            return -1;
    }

    /* Prefetch srcData of previous layer */
    if (vdnnMode != BASELINE && _layer_id_to_prefetch >= 0)
        prefetchPreviousSrcData(_prefetchedSrcData_h, _srcDataToPrefetch, _prefetch_bytes);

    const DATATYPE alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnSoftmaxBackward(
        *cudnnHandle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha,
        dstTensorDesc, dstData,
        dstDiffTensorDesc, diffData,
        &beta,
        srcDiffTensorDesc, gradData
    ));

    return 0;
}

int Layer::poolingForward(bool* _offloaded, DATATYPE** _offloadedSrcData_h, void* workSpace, bool isDynamic) {
    /* Allocate GPU memory for dstData */
    if (vdnnMode != BASELINE) {
        if (cnmemMallocDstData(isDynamic) < 0)
            return -1;
    }
    
    const DATATYPE alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnPoolingForward(
        *cudnnHandle,
        poolingDesc,
        &alpha,
        srcTensorDesc, srcData,
        &beta,
        dstTensorDesc, dstData
    ));

    /* Offload srcData to host */
    if (_offloaded[layerId])
        offloadSrcData(_offloaded, _offloadedSrcData_h);

    return 0;
}

int Layer::poolingBackward(int _layer_id_to_prefetch, DATATYPE* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes, void* workSpace, bool isDynamic) {
    /* Allocate GPU memory for gradData */
    if (vdnnMode != BASELINE) {
        if (cnmemMallocGradData(isDynamic) < 0)
            return -1;
    }

    /* Prefetch srcData of previous layer */
    if (vdnnMode != BASELINE && _layer_id_to_prefetch >= 0)
        prefetchPreviousSrcData(_prefetchedSrcData_h, _srcDataToPrefetch, _prefetch_bytes);

    const DATATYPE alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnPoolingBackward(
        *cudnnHandle,
        poolingDesc,
        &alpha,
        dstTensorDesc, dstData,
        dstDiffTensorDesc, diffData,
        srcTensorDesc, srcData,
        &beta,
        srcDiffTensorDesc, gradData
    ));

    return 0;
}

int Layer::concatenateForward() {
    return 0;
}

int Layer::concatenateBackward() {
    return 0;
}

void Layer::initializeConvolutionFilterAndBias() {
    /* Random initialization of filter and bias data using
       randomly generated DATATYPE number between -1 and 1 */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<DATATYPE> dis(-RANDOM_VALUE_LIMIT, RANDOM_VALUE_LIMIT);

    int filterDataSize = FilterDataSize();
    int biasDataSize = c_out;
    DATATYPE* filterData_h = new DATATYPE[filterDataSize];
    DATATYPE* biasData_h = new DATATYPE[biasDataSize];
    for (int i = 0; i < filterDataSize; i++) { filterData_h[i] = dis(gen); }
    for (int i = 0; i < biasDataSize; i++) { biasData_h[i] = dis(gen); }

    ASSERT_EQ(cnmemMalloc((void**) &filterData, sizeof(DATATYPE) * filterDataSize, NULL));
    checkCudaErrors(cudaMemcpy(filterData, filterData_h, sizeof(DATATYPE) * filterDataSize, cudaMemcpyHostToDevice));
    ASSERT_EQ(cnmemMalloc((void**) &biasData, sizeof(DATATYPE) * biasDataSize, NULL));
    checkCudaErrors(cudaMemcpy(biasData, biasData_h, sizeof(DATATYPE) * biasDataSize, cudaMemcpyHostToDevice));
    delete filterData_h;
    delete biasData_h;
}

void Layer::initializeFullyConnectedFilterAndBias() {
    /* Random initialization of filter and bias data using
       randomly generated DATATYPE number between -1 and 1 */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<DATATYPE> dis(-RANDOM_VALUE_LIMIT, RANDOM_VALUE_LIMIT);

    int filterDataSize = c_in * c_out;
    int biasDataSize = c_out;
    DATATYPE* filterData_h = new DATATYPE[filterDataSize];
    DATATYPE* biasData_h = new DATATYPE[biasDataSize];
    for (int i = 0; i < filterDataSize; i++) { filterData_h[i] = dis(gen); }
    for (int i = 0; i < biasDataSize; i++) { biasData_h[i] = dis(gen); }

    ASSERT_EQ(cnmemMalloc((void**) &filterData, sizeof(DATATYPE) * filterDataSize, NULL));
    checkCudaErrors(cudaMemcpy(filterData, filterData_h, sizeof(DATATYPE) * filterDataSize, cudaMemcpyHostToDevice));
    ASSERT_EQ(cnmemMalloc((void**) &biasData, sizeof(DATATYPE) * biasDataSize, NULL));
    checkCudaErrors(cudaMemcpy(biasData, biasData_h, sizeof(DATATYPE) * biasDataSize, cudaMemcpyHostToDevice));
    delete filterData_h;
    delete biasData_h;
}

void Layer::offloadSrcData(bool* _offloaded, DATATYPE** _offloadedSrcData_h) {
    checkCuda(cudaMemcpyAsync(
        _offloadedSrcData_h[layerId], 
        srcData, 
        SrcDataSize() * sizeof(DATATYPE), 
        cudaMemcpyDeviceToHost,
        *stream_memory
    ));
}

void Layer::prefetchPreviousSrcData(DATATYPE* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes) {
    checkCuda(cudaMemcpyAsync(
        _srcDataToPrefetch, 
        _prefetchedSrcData_h, 
        _prefetch_bytes, 
        cudaMemcpyHostToDevice,
        *stream_memory
    ));
}

int Layer::cnmemMallocSrcData(bool isDynamic) {
    if (ASSERT_EQ(cnmemMalloc((void**) &srcData, sizeof(DATATYPE) * SrcDataSize(), NULL), isDynamic) < 0)
        return -1;
    return 0;
}

int Layer::cnmemMallocDstData(bool isDynamic) {
    if (ASSERT_EQ(cnmemMalloc((void**) &dstData, sizeof(DATATYPE) * DstDataSize(), NULL), isDynamic) < 0)
        return -1;
    return 0;
}

int Layer::cnmemMallocDiffData(bool isDynamic) {
    if (ASSERT_EQ(cnmemMalloc((void**) &diffData, sizeof(DATATYPE) * SrcDataSize(), NULL), isDynamic) < 0)
        return -1;
    return 0;
}

int Layer::cnmemMallocGradData(bool isDynamic) {
    if (ASSERT_EQ(cnmemMalloc((void**) &gradData, sizeof(DATATYPE) * SrcDataSize(), NULL), isDynamic) < 0)
        return -1;
    return 0;
}

void Layer::findMemOptFwdAlgo() {
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
        *cudnnHandle,
        srcTensorDesc,
        filterDesc,
        convDesc,
        dstTensorDesc,
        CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
        0,
        &fwdAlgo
    ));
}

void Layer::findMemOptBwdFilterAlgo() {
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
        *cudnnHandle,
        srcTensorDesc,
        dstDiffTensorDesc,
        convDesc,
        filterDesc, // Descriptor for filter diff tensor
        CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,
        0,
        &bwdFilterAlgo
    ));
}

void Layer::findMemOptBwdDataAlgo() {
    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
        *cudnnHandle,
        filterDesc,
        dstDiffTensorDesc,
        convDesc,
        srcDiffTensorDesc,
        CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
        0,
        &bwdDataAlgo
    ));
}

void Layer::findPerfOptFwdAlgo() {
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
        *cudnnHandle,
        srcTensorDesc,
        filterDesc,
        convDesc,
        dstTensorDesc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &fwdAlgo
    ));
}

void Layer::findPerfOptBwdFilterAlgo() {
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
        *cudnnHandle,
        srcTensorDesc,
        dstDiffTensorDesc,
        convDesc,
        filterDesc, // Descriptor for filter diff tensor
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        0,
        &bwdFilterAlgo
    ));
}

void Layer::findPerfOptBwdDataAlgo() {
    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
        *cudnnHandle,
        filterDesc,
        dstDiffTensorDesc,
        convDesc,
        srcDiffTensorDesc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &bwdDataAlgo
    ));
}

void Layer::findDynamicAlgo() {
    void* dummy = NULL;

    /* Fwd Algorithm */
    for (int i = NUM_FWD_ALGO; i >= 0; i--) {
        fwdAlgo = fwdAlgoList[i];
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            *cudnnHandle,
            srcTensorDesc,
            filterDesc,
            convDesc,
            dstTensorDesc,
            fwdAlgo,
            &fwdWorkSpaceSize
        ));
        if (ASSERT_EQ(cnmemMalloc((void**) &dummy, fwdWorkSpaceSize, NULL), true) >= 0) {
            ASSERT_EQ(cnmemFree(dummy, NULL), false);
            break;
        }
    }

    /* Bwd Data Algorithm */
    for (int i = NUM_BWD_DATA_ALGO; i >= 0; i--) {
        bwdDataAlgo = bwdDataAlgoList[i];
        checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
            *cudnnHandle,
            filterDesc,
            dstDiffTensorDesc,
            convDesc,
            srcDiffTensorDesc,
            bwdDataAlgo,
            &bwdDataWorkSpaceSize
        ));
        if (ASSERT_EQ(cnmemMalloc((void**) &dummy, bwdDataWorkSpaceSize, NULL), true) >= 0) {
            ASSERT_EQ(cnmemFree(dummy, NULL), false);
            break;
        }
    }

    /* Bwd Filter Algorithm */
    for (int i = NUM_BWD_FILTER_ALGO; i >= 0; i--) {
        bwdFilterAlgo = bwdFilterAlgoList[i];
        checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            *cudnnHandle,
            srcTensorDesc,
            dstDiffTensorDesc,
            convDesc,
            filterDesc,
            bwdFilterAlgo,
            &bwdFilterWorkSpaceSize
        ));
        if (ASSERT_EQ(cnmemMalloc((void**) &dummy, bwdFilterWorkSpaceSize, NULL), true) >= 0) {
            ASSERT_EQ(cnmemFree(dummy, NULL), false);
            break;
        }
    }
}

void Layer::printDstData() {
    printf("Layer ID %d dstData\n", layerId);
    DATATYPE* dstData_h = new DATATYPE[DstDataSize()]; 
    checkCudaErrors(cudaMemcpy(dstData_h, dstData, sizeof(DATATYPE) * DstDataSize(), cudaMemcpyDeviceToHost));
    for (int n = 0; n < n_out; n++) {
        for (int c = 0; c < c_out; c++) {
            for (int h = 0; h < h_out; h++) {
                for (int w = 0; w < w_out; w++) {
                    printf("%f ", dstData_h[n * (c_out * h_out * w_out) + c * (h_out * w_out) + h * h_out + w]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
    delete dstData_h;
}

void Layer::printFilterData() {
    printf("Layer ID %d filterData\n", layerId);
    DATATYPE* filterData_h = new DATATYPE[FilterDataSize()]; 
    checkCudaErrors(cudaMemcpy(filterData_h, filterData, sizeof(DATATYPE) * FilterDataSize(), cudaMemcpyDeviceToHost));
    for (int _k = 0; _k < k; _k++) {
        for (int _c = 0; _c < c_in; _c++) {
            for (int _r = 0; _r < r; _r++) {
                for (int _s = 0; _s < s; _s++) {
                    printf("%f ", filterData_h[_k * (c_in * r * s) + _c * (r * s) + _r * r + _s]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
    delete filterData_h;
}