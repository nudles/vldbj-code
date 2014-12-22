#!/usr/bin/env sh

./build/tools/caffe train --gpu=2 --solver=models/singlelabel/text_solver.prototxt 
