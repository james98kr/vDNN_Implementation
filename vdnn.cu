#include "vdnn.h"
#include "network.h"
#include "layer.h"

using namespace std;
using namespace std::chrono;

void parseVdnnInputs(char** argv, CNN_t* _cnnType, vdnn_t* _vdnnType, vdnnAlgoMode_t* _algoMode) {
    /* Set CNN model */
    if (strcmp(argv[1], "ALEXNET") == 0)
        *_cnnType = ALEXNET;
    else if (strcmp(argv[1], "OVERFEAT") == 0)
        *_cnnType = OVERFEAT;
    else if (strcmp(argv[1], "GOOGLENET") == 0)
        *_cnnType = GOOGLENET;
    else if (strcmp(argv[1], "VGG16_64") == 0)
        *_cnnType = VGG16_64;
    else if (strcmp(argv[1], "VGG16_128") == 0)
        *_cnnType = VGG16_128;
    else if (strcmp(argv[1], "VGG16_256") == 0)
        *_cnnType = VGG16_256;
    else if (strcmp(argv[1], "CUSTOM_CNN") == 0)
        *_cnnType = CUSTOM_CNN;
    else
        M_ASSERT(0, "CNN type must be one of ALEXNET, OVERFEAT, GOOGLENET, VGG16_64, VGG16_128, VGG16_256, CUSTOM_CNN");

    /* Set VDNN Type */
    if (strcmp(argv[2], "BASELINE") == 0)
        *_vdnnType = BASELINE;
    else if (strcmp(argv[2], "VDNN_NONE") == 0)
        *_vdnnType = VDNN_NONE;
    else if (strcmp(argv[2], "VDNN_ALL") == 0)
        *_vdnnType = VDNN_ALL;
    else if (strcmp(argv[2], "VDNN_CONV") == 0)
        *_vdnnType = VDNN_CONV;
    else
        M_ASSERT(0, "VDNN type must be one of BASELINE, VDNN_NONE, VDNN_ALL, VDNN_CONV");

    /* Set Algorithm Type */
    if (strcmp(argv[3], "VDNN_MEMORY_OPT_ALGO") == 0)
        *_algoMode = VDNN_MEMORY_OPT_ALGO;
    else if (strcmp(argv[3], "VDNN_PERF_OPT_ALGO") == 0)
        *_algoMode = VDNN_PERF_OPT_ALGO;
    else if (strcmp(argv[3], "VDNN_DYNAMIC_ALGO") == 0)
        *_algoMode = VDNN_DYNAMIC_ALGO;
    else
        M_ASSERT(0, "Algorithm type must be one of VDNN_MEMORY_OPT_ALGO, VDNN_PERF_OPT_ALGO, VDNN_DYNAMIC_ALGO");

    printf("################# VDNN Configurations #################\n");
    printf("CNN Model Type: %s\n", argv[1]);
    printf("VDNN Mode: %s\n", argv[2]);
    printf("Convolution Algorithm: %s\n", argv[3]);
    printf("#######################################################\n\n");
}

int main(int argc, char** argv) {
    /* Set default device, and check argc is 4 */
    M_ASSERT(argc == 4, "Need to specify CNN model, VDNN mode, and Algorithm choice");
    checkCuda(cudaSetDevice(DEFAULT_DEVICE));
    cudaDeviceReset();

    /* Parse input into cnnType, vdnnType, algoMode */
    CNN_t cnnType;
    vdnn_t vdnnType;
    vdnnAlgoMode_t algoMode;
    parseVdnnInputs(argv, &cnnType, &vdnnType, &algoMode);

    /* Initialize network */
    Network network(cnnType, vdnnType, algoMode);

    /* Perform one iteration of fwd/bwd propagation */
    auto t1 = high_resolution_clock::now();
    network.forwardPropagation();
    network.backwardPropagation();
    auto t2 = high_resolution_clock::now();

    /* Print results */
    auto duration = duration_cast<microseconds>(t2 - t1);
    cout << "Elapsed Time: " << duration.count() << "ms" << endl;
    cout << "" << endl;

    return 0;
}