#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

NUSWIDE=models/nuswide
DATA=../..//data/nuswide/wordvec
IMAGE_ROOT=../../data/nuswide/images
TOOLS=build/tools

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$IMAGE_ROOT" ]; then
  echo "Error: IMAGE_ROOT is not a path to a directory: $IMAGE_ROOT"
  echo "Set the IMAGE_ROOT variable in create_imagenet.sh to the path" \
       "where the images are stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_nuswide \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $IMAGE_ROOT \
    $DATA/trainvec.dat \
    $NUSWIDE/nuswide_train_lmdb

echo 'spliting testset into val and test'
split -n l/2 $DATA/testvec.dat $DATA/split

echo "Creating val lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_nuswide \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $IMAGE_ROOT \
    $DATA/splitaa \
    $NUSWIDE/nuswide_val_lmdb

echo "Creating test lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_nuswide \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $IMAGE_ROOT \
    $DATA/splitab \
    $NUSWIDE/nuswide_test_lmdb

echo "Done."
