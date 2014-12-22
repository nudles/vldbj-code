#!/usr/bin/env sh

./build/tools/caffe --gpu=2 train --solver=models/singlelabel/image_solver.prototxt \
    --weights=models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
