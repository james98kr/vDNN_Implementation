#ifndef _LAYER_H_
#define _LAYER_H_

#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <assert.h>

#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#include <cublas_v2.h>
#include <cudnn.h>

#include "vdnn.h"
#include "./cnmem/include/cnmem.h"

#include <unistd.h>
#include <time.h>
#include <pthread.h>

#ifdef USE_CPP_11
#include <thread>
#endif

/* Macro related to kernel execution */
#define CEIL(a, b) ((a + b - 1) / b)

/* Profiling individual layers */
#define _PROFILE_INDIVIDUAL_LAYERS_ 1
#define _PROFILE_FWD_ 1 
#define _PROFILE_BWD_ 0
#define _PROFILE_LAYER_ID_ 999

/* PCIe */
// PCI-E (gen2) --  8 GB/s (with 80% effective bw)
// PCI-E (gen3) -- 16 GB/s (with 80% effective bw)
// Set PCIE_BANDWIDTH to 8 or 16
#define PCIE_BANDWIDTH 8
#define OFFLOAD_PREFETCH_BW ((float)1024 * 1024 * PCIE_BANDWIDTH * 0.8)

/* Not sure what these are for */
#define _REUSE_DISTANCE_ 1
#define _USE_CNMEM_MEM_USAGE_ 1
#define _INCLUDE_FC_LAYERS_ 1
#define _COMPUTE_CHECKSUM_ 1
#ifdef _COMPUTE_CHECKSUM_
#include <zlib.h>
#else
#define _PROFILE_CONV_ALGO_ 1
#endif

/* Some debugging tools for CUDA, cuDNN, cuBLAS
   Useful wrappers for any function in the above libraries */
#define M_ASSERT(EXPR, MESSAGE) {                                      \
    assert(((void)(MESSAGE), (EXPR)));                                 \
}

#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << status;                           \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCUBLAS(status) {                                          \
    std::stringstream _error;                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                             \
      _error << "CUBLAS failure: " << status;                          \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      assert(0);                                                       \
      FatalError(_error.str());                                        \
    }                                                                  \
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

inline
int ASSERT_EQ(cnmemStatus_t result, bool isDynamicMode = false) {
    if (!isDynamicMode) {
        if (result != CNMEM_STATUS_SUCCESS) { 
            printf("\n[CNMEM FAILED with error %s]\n", cnmemGetErrorString(result));
            assert(0); 
        }
        return 0;
    }
    else {
        if (result == CNMEM_STATUS_OUT_OF_MEMORY)
            return -1;
        else if (result == CNMEM_STATUS_SUCCESS)
            return 0;
        else
            return -2;
    }
    return 0; // should not reach here
}

class Layer {

public:
    Layer(
        vdnn_t          _vdnnMode,
        vdnnAlgoMode_t  _vdnnAlgoMode,
        Layer_t         _layerType,
        cudnnHandle_t*  _cudnnHandle,
        cudaStream_t*   _stream_compute,
        cudaStream_t*   _stream_memory,
        int _id,
        int _n, int _c, int _h, int _w, 
        int _pad_h, int _pad_w, int _stride_h, int _stride_w, 
        int _k, int _r, int _s);
    Layer(
        vdnn_t          _vdnnMode,
        vdnnAlgoMode_t  _vdnnAlgoMode,
        Layer_t         _layerType,
        cublasHandle_t* _cublasHandle,
        cudaStream_t*   _stream_compute,
        cudaStream_t*   _stream_memory,
        int _id,
        int _n, int _c, int _h, int _w, 
        int _pad_h, int _pad_w, int _stride_h, int _stride_w, 
        int _k, int _r, int _s);
    ~Layer(); 

    // Randomly initialize filter/bias
    void    initializeConvolutionFilterAndBias();
    void    initializeFullyConnectedFilterAndBias();

    // Set fwd/bwd algo to memory-optimal or perf-optimal algorithm, or find algo dynamically
    void    findMemOptFwdAlgo();
    void    findMemOptBwdFilterAlgo();
    void    findMemOptBwdDataAlgo();
    void    findPerfOptFwdAlgo();
    void    findPerfOptBwdFilterAlgo();
    void    findPerfOptBwdDataAlgo();
    void    findDynamicAlgo();
  
    // General fwd/bwd function
    int     forward(bool* _offloaded, DATATYPE** _offloadedSrcData_h, void* workSpace, bool isDynamic);
    int     backward(int _layer_id_to_prefetch, DATATYPE* _prefetchedSrcData_h, 
                   void* _srcDataToPrefetch, unsigned long _prefetch_bytes, void* workSpace, bool isDynamic);

    // CONV layer
    int     convolutionForward(bool* _offloaded, DATATYPE** _offloadedSrcData_h, void* workSpace, bool isDynamic);
    int     convolutionBackward(int _layer_id_to_prefetch, DATATYPE* _prefetchedSrcData_h, 
                              void* _srcDataToPrefetch, unsigned long _prefetch_bytes, void* workSpace, bool isDynamic);

    // FC_GEMM layer
    int     fullyConnectedForward(bool* _offloaded, DATATYPE** _offloadedSrcData_h, void* workSpace, bool isDynamic);
    int     fullyConnectedBackward(int _layer_id_to_prefetch, DATATYPE* _prefetchedSrcData_h, 
                                 void* _srcDataToPrefetch, unsigned long _prefetch_bytes, void* workSpace, bool isDynamic);

    // Activation(RELU, TANH, SIGMOID) layer
    int     activationForward(bool* _offloaded, DATATYPE** _offloadedSrcData_h, void* workSpace, bool isDynamic);
    int     activationBackward(int _layer_id_to_prefetch, DATATYPE* _prefetchedSrcData_h, 
                             void* _srcDataToPrefetch, unsigned long _prefetch_bytes, void* workSpace, bool isDynamic);

    // SOFTMAX layer
    int     softmaxForward(bool* _offloaded, DATATYPE** _offloadedSrcData_h, void* workSpace, bool isDynamic);
    int     softmaxBackward(int _layer_id_to_prefetch, DATATYPE* _prefetchedSrcData_h, 
                          void* _srcDataToPrefetch, unsigned long _prefetch_bytes, void* workSpace, bool isDynamic);

    // POOL layer
    int     poolingForward(bool* _offloaded, DATATYPE** _offloadedSrcData_h, void* workSpace, bool isDynamic);
    int     poolingBackward(int _layer_id_to_prefetch, DATATYPE* _prefetchedSrcData_h, 
                          void* _srcDataToPrefetch, unsigned long _prefetch_bytes, void* workSpace, bool isDynamic);

    // CONCATENATE layer
    int     concatenateForward();
    int     concatenateBackward();

    // Offloading and prefetching functions
    void    offloadSrcData(bool* _offloaded, DATATYPE** _offloadedSrcData_h);
    void    prefetchPreviousSrcData(DATATYPE* _prefetchedSrcData_h, void* _srcDataToPrefetch, unsigned long _prefetch_bytes);

    // Allocate device memory
    int     cnmemMallocSrcData(bool isDynamic);
    int     cnmemMallocDstData(bool isDynamic);
    int     cnmemMallocDiffData(bool isDynamic);
    int     cnmemMallocGradData(bool isDynamic);

    // Copy pointers 
    void    copySrcData(void* _srcData)   { srcData = _srcData;                     }
    void    copyDstData(void* _dstData)   { dstData = _dstData;                     }
    void    copyDiffData(void* _diffData) { diffData = _diffData;                   }
    void    copyGradData(void* _gradData) { gradData = _gradData;                   }

    /* Set functions */
    void    setVdnnMode(vdnn_t _vdnnMode)                                    { vdnnMode = _vdnnMode;           }
    void    setFwdAlgo(cudnnConvolutionFwdAlgo_t _fwdAlgo)                   { fwdAlgo = _fwdAlgo;             }
    void    setBwdDataAlgo(cudnnConvolutionBwdDataAlgo_t _bwdDataAlgo)       { bwdDataAlgo = _bwdDataAlgo;     }
    void    setBwdFilterAlgo(cudnnConvolutionBwdFilterAlgo_t _bwdFilterAlgo) { bwdFilterAlgo = _bwdFilterAlgo; }

    /* Get functions */
    Layer_t LayerType()              { assert(layerType<NUM_LAYER_TYPES); return layerType; }
    void*   SrcData()                { return srcData;                                      }
    void*   DstData()                { return dstData;                                      }
    void*   DiffData()               { return diffData;                                     }
    void*   GradData()               { return gradData;                                     }
    int     InputN()                 { return n_in;                                         }
    int     InputC()                 { return c_in;                                         }
    int     InputH()                 { return h_in;                                         }
    int     InputW()                 { return w_in;                                         }
    int     OutputN()                { return n_out;                                        }
    int     OutputC()                { return c_out;                                        }
    int     OutputH()                { return h_out;                                        }
    int     OutputW()                { return w_out;                                        }
    int     SrcDataSize()            { return n_in * c_in * h_in * w_in;                    }
    int     DstDataSize()            { return n_out * c_out * h_out * w_out;                }
    size_t  FwdWorkSpaceSize()       { return fwdWorkSpaceSize;                             }
    size_t  BwdDataWorkSpaceSize()   { return bwdDataWorkSpaceSize;                         }
    size_t  BwdFilterWorkSpaceSize() { return bwdFilterWorkSpaceSize;                       }
    int     FilterDataSize()         { return k * c_in * r * s;                             }
    int     RefCntFwd()              { return refCntFwd;                                    }
    int     RefCntBwd()              { return refCntBwd;                                    }

    /* Reference Count Management */
    void    decrementRefCntFwd()  { assert(refCntFwd>0);  refCntFwd--;              }
    void    decrementRefCntBwd()  { assert(refCntBwd>0);  refCntBwd--;              }

    /* For debugging */
    void    printDstData();
    void    printFilterData();

private:
    /* General layer info */
    vdnn_t         vdnnMode;
    vdnnAlgoMode_t vdnnAlgoMode;
    Layer_t        layerType;
    int            layerId;

    /* Input NCHW */
    int       n_in;
    int       c_in;
    int       h_in;
    int       w_in;

    /* Filter info */
    int       pad_h;
    int       pad_w;
    int       stride_h;
    int       stride_w;
    int       k;
    int       r;
    int       s;

    /* Output NCHW */
    int       n_out;
    int       c_out;
    int       h_out;
    int       w_out;

    /* Workspace size */
    size_t    fwdWorkSpaceSize;
    size_t    bwdDataWorkSpaceSize;
    size_t    bwdFilterWorkSpaceSize;

    /* Streams */
    cudaStream_t* 	  stream_compute;
    cudaStream_t*	  stream_memory;

    /* Pointers to memory (data) */
    void*     srcData;
    void*     filterData;
    void*     biasData;
    void*     dstData;
    void*     diffData;
    void*     gradData;

    /* Connection to other layers */
    std::vector<int>  producerLayerId;
    std::vector<int>  consumerLayerId;
    int               refCntFwd;
    int               refCntBwd;

    /*************/
    /*   cuDNN   */
    /*************/
    cudnnHandle_t*                        cudnnHandle; // shared across all layers
    cudnnTensorDescriptor_t               srcTensorDesc, dstTensorDesc, biasTensorDesc;
    cudnnTensorDescriptor_t               srcDiffTensorDesc, dstDiffTensorDesc;
    cudnnFilterDescriptor_t               filterDesc;
    cudnnConvolutionDescriptor_t          convDesc;
    cudnnPoolingDescriptor_t              poolingDesc;
    cudnnActivationDescriptor_t	          actvDesc;
    bool                                  inPlaceOp;    

    // Algorithms for forward/backward propagation
    cudnnConvolutionFwdAlgo_t		      fwdAlgo;
    cudnnConvolutionBwdDataAlgo_t         bwdDataAlgo;
    cudnnConvolutionBwdFilterAlgo_t	      bwdFilterAlgo;

    // Performance results (profile)
    cudnnConvolutionFwdAlgoPerf_t*        fwdProfileResults;
    cudnnConvolutionBwdFilterAlgoPerf_t*  bwdFilterProfileResults;
    cudnnConvolutionBwdDataAlgoPerf_t*    bwdDataProfileResults;
    int                                   profiledFwdAlgoCount;
    int                                   profiledBwdFilterAlgoCount;
    int                                   profiledBwdDataAlgoCount;

    /**************/
    /*   cuBLAS   */
    /**************/
    cublasHandle_t*                       cublasHandle; // shared across all layers
    bool                                  insideInceptionModule;
    bool                                  headOfFork;
    bool                                  tailOfFork;
    int                                   forkId;
    int                                   concatenateChannelOffset;
};

#endif