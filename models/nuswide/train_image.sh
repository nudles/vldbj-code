#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/nuswide/image_solver.prototxt \
    --weights=models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
