#/usr/bin/bash
nvcc -g vdnn.cu network.cu layer.cu -lcudnn -lcublas -lcnmem -o vdnn