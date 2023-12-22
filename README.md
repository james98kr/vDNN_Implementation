# vDNN: Virtualized Deep Neural Network Implementation
This repository holds CUDA/cuDNN implementation of vDNN, proposed in the paper [**vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design**](https://arxiv.org/abs/1602.08124). 

## Installation & Setup
1\) Clone repository
```
git clone https://github.com/james98kr/vDNN_Implementation.git
cd vDNN_Implementation
```

2\) Compile source code into binary executable using the following command. This will create the executable file ``vdnn``. 
```
./build.sh
```

## Execution

Execute the file using the following command:
```
./vdnn {CNN_MODEL} {VDNN_TYPE} {ALGO_MODE}
```

The following are the types of CNN models supported:
* ALEXNET
* OVERFEAT
* VGG16_64
* VGG16_128

The following are the types of VDNN policies supported:
* BASELINE
    * Baseline policy
    * No offloading or prefetching of ``srcData`` in any of the layers
    * ``diffData``, ``gradData``, and ``workSpace`` are reused throughput propagation
* VDNN_ALL
    * vDNN_all policy described in paper
    * `srcData` of all layers are offloaded and later prefetched to host memory
    * `diffData`, `gradData`, and `workSpace` of all layers are allocated separately
* VDNN_CONV
    * vDNN_conv policy described in paper
    * `srcData` of only convolution layers are offloaded and later prefetched to host memory
    * `diffData`, `gradData`, and `workSpace` of all layers are allocated separately
* VDNN_NONE
    * Used for reference
    * No offloading or prefetching of `srcData` in any of the layers
    * `diffData`, `gradData`, and `workSpace` of all layers are allocated separately

The following are the types of convolution algorithm modes supported:
* VDNN_MEMORY_OPT_ALGO
    * Set all fwd/bwd propagation algorithms of all `CONV` layers to memory-optimal algorithm
    * No workspace is required to execute forward/backward propagation of `CONV` layers
    * In cuDNN version 7.65, this is `IMPLICIT_GEMM` algorithm
* VDNN_PERF_OPT_ALGO
    * Set all fwd/bwd propagation algorithms of all `CONV` layers to performance-optimal algorithm
    * Some workspace is required to execute forward/backward propagation of `CONV` layers

Thus, you can combine the above options to execute vDNN. For instance, a few examples would be:
```
./vdnn OVERFEAT BASELINE VDNN_MEMORY_OPT_ALGO
./vdnn VGG16_128 VDNN_ALL VDNN_MEMORY_OPT_ALGO
./vdnn ALEXNET VDNN_NONE VDNN_PERF_OPT_ALGO
```