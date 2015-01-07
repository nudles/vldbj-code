./build/tools/caffe train --gpu=2  \
  --solver=models/nuswide/multilabel/joint_solver.prototxt \
  --weights=data/nuswide/multilabel/snapshot/image-path_iter_50000.caffemodel\
  --weights2=data/nuswide/multilabel/snapshot/text-path_iter_40000.caffemodel
