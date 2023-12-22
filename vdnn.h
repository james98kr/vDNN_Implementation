#ifndef _VDNN_H_
#define _VDNN_H_

/******************* Parameters *******************/
#define DEFAULT_DEVICE 0
#define DEFAULT_NUM_LAYER 40
#define DEFAULT_BATCH_SIZE 128
#define LEARNING_RATE 1e-5
#define BLOCK_SIZE 256
#define RANDOM_VALUE_LIMIT 1e-2
#define DATATYPE float
#define CUDNN_DATATYPE CUDNN_DATA_FLOAT
/******************* Parameters *******************/

/***********************FIXED**********************/
#define NUM_FWD_ALGO 8
#define NUM_BWD_DATA_ALGO 6
#define NUM_BWD_FILTER_ALGO 5
#define CUDNN_TENSOR_FORMAT CUDNN_TENSOR_NCHW
#define CUDNN_CONV_MODE CUDNN_CROSS_CORRELATION
/***********************FIXED**********************/

typedef enum
{
    BASELINE      = 0,
    VDNN_NONE     = 1,
    VDNN_ALL      = 2,
    VDNN_CONV     = 3,
} vdnn_t;

typedef enum
{
    VDNN_MEMORY_OPT_ALGO  = 0,
    VDNN_PERF_OPT_ALGO    = 1,
    VDNN_DYNAMIC_ALGO     = 2,
} vdnnAlgoMode_t;

typedef enum
{
    FC_GEMM         = 0,
    CONV            = 1,
    FC_CONV         = 2,
    RELU            = 3,
    TANH            = 4,
    SIGMOID         = 5,
    SOFTMAX         = 6,
    POOL            = 7,
    CONCATENATE     = 8,
    NUM_LAYER_TYPES = 9,
} Layer_t;

typedef enum
{
    ALEXNET         = 0,
    OVERFEAT        = 1, // Accurate model
    GOOGLENET       = 2,
    VGG16_64        = 3,
    VGG16_128       = 4,
    VGG16_256       = 5,
    CUSTOM_CNN      = 6,
} CNN_t;

#endif