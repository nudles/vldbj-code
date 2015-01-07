./build/tools/caffe train --gpu=2  \
  --solver=models/nuswide/singlelabel/joint_solver.prototxt \
  --weights=data/nuswide/singlelabel/snapshot/image-path_iter_40000.caffemodel\
  --weights2=data/nuswide/singlelabel/snapshot/text-path_iter_40000.caffemodel
