#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <assert.h>
#include <string.h>
#include <cuda.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include "layer.h"

class Network {

public:
    Network(        
        CNN_t		    _cnnType,
        vdnn_t		    _vdnnType,
        vdnnAlgoMode_t	_algoMode,
        int		        _batchSize = DEFAULT_BATCH_SIZE);
    ~Network();

    /* Reset network */
    void            reset(CNN_t _cnnType, vdnn_t _vdnnType, vdnnAlgoMode_t _algoMode);

    /* Setup */
    void            setLayers(CNN_t _cnnType);
    void            setLayersAlexnet();
    void            setLayersOverfeat();
    void            setLayersGooglenet();
    void            setLayersVgg16(int givenBatch);
    void            setLayersCustomCNN();
    void            randomInitializeData(void* data, int size);
    void            sanityCheck();
    void            setVdnnType(vdnn_t newVdnnType);
    void            setAlgoType(vdnnAlgoMode_t newAlgoMode);//todo
    void            vdnnDynamicAlgo();

    /* Find max dx, dy, workspace size */
    int             findMaxDxDySize();
    int             findMaxWorkSpaceSize();

    /* Computation for training network */
    void            calculateFirstDiffData(void* result, void* first, void* second);
    int             forwardPropagation(bool isDynamic = false);
    int             backwardPropagation(bool isDynamic = false);

    /* Find prefetch layer */
    void            resetIsPrefetchedArray();
    int             findPrefetchLayer(int currLayerId);

    /* Get functions */
    Layer**         layerArray()            { return layer;                         }
    DATATYPE**      offloadedSrcDataArray() { return offloadedSrcData_h;            }
    int             getInputSize()          { return n_in * c_in * h_in * w_in;     }
    int             getOutputSize()         { return n_out * c_out * h_out * w_out; }

private:
    CNN_t	            cnnType;
    vdnn_t              vdnnType;
    vdnnAlgoMode_t      algoMode;
    cudnnHandle_t       cudnnHandle;
    cublasHandle_t      cublasHandle;
    cudaStream_t        stream_compute;
    cudaStream_t        stream_memory;
    int                 batchSize;
    int                 numLayers;

    /* Input */
    int	                n_in;
    int	                c_in;
    int	                h_in;
    int	                w_in;

    /* Output */
    int	                n_out;
    int	                c_out;
    int	                h_out;
    int	                w_out;

    /* Data */
    void*               inputData;
    void*               labelData;
    void*               networkDiffData;
    void*               networkGradData;
    void*               workSpace;

    /* Layers */
    Layer*              layer[DEFAULT_NUM_LAYER];
    bool                isOffload[DEFAULT_NUM_LAYER];
    bool                isPrefetched[DEFAULT_NUM_LAYER];
    DATATYPE*           offloadedSrcData_h[DEFAULT_NUM_LAYER];
};

#endif
