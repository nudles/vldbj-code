#!/usr/bin/env sh

./build/tools/caffe train --gpu=2 --solver=models/nuswide/singlelabel/text_solver.prototxt 
