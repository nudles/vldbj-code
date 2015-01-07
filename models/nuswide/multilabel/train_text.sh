#!/usr/bin/env sh

./build/tools/caffe train --gpu=2 --solver=models/nuswide/multilabel/text_solver.prototxt 
