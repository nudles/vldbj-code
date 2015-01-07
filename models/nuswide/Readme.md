All operations are executed under the top directory of caffe
1. Dataset Preprocessing:
  download the NUSWIDE dataset into data/nuswide/raw_input
  after unzip the zip files, you should have the following file (folder)s in that folder:
       AllLabels/  AllTags1k.txt All_Tags.txt  Imagelist.txt  images/

1.1. extract valid(with at least one label and one tag) records from NUSWIDE dataset. 
  $python models/nuswide/preprocess/extract_records.py -l data/nuswide/raw_input/AllLabels -t data/nuswide/raw_input/AllTags1k.txt -f data/nuswide/raw_input/All_Tags.txt -i data/nuswide/raw_input/Imagelist.txt -d data/nuswide/raw_input/images/ -o data/nuswide/input

  you should see the similar output as follows:
  2014-12-22 13:16:36,649 - root - INFO - There are 269648 labels, 81 images
  2014-12-22 13:17:41,417 - root - INFO - max labels assocaited to one image is 13
  2014-12-22 13:17:41,504 - root - INFO - There are 209347 images with at least one label 
  2014-12-22 13:20:05,030 - root - INFO - There are 261033 images with at least one tag 
  2014-12-22 13:20:05,046 - root - INFO - There are 203598 images with at least one label and tag 
  2014-12-22 13:20:05,047 - root - INFO - random shuffle index
  2014-12-22 13:20:05,771 - root - INFO - Writing records into data/nuswide/input/record-tagwords.dat
  2014-12-22 13:20:05,771 - root - INFO - recordlist.txt:<index in the original image list> <relative path to image> [label id among 81 labels in the alllabel.txt file]#$$#[tags ]

  2014-12-22 13:20:09,696 - root - INFO - Check images from data/nuswide/input/record-tagwords.dat
  2014-12-22 13:20:11,021 - root - INFO - There are 203598 images in total, 0 missing images
  2014-12-22 13:20:11,021 - root - INFO - Finished Preprocessing

  the output file (record-tagwords.dat) contains record id, image path, label 
  indexes and tag words. it also generates a label dictionary (label-dict.txt) 
  from label id to label name.

1.2. train word2vec model and convert tags to word vectors
  train the word2vec model using records extracted from step 1.
  $python models/nuswide/preprocess/myword2vec.py  --model data/nuswide/input/word2vec.model  --input data/nuswide/input/record-tagwords.dat  --ntrain 150000 --n 100

  you can check the learned vectors by finding the most similar words:
  $python models/nuswide/preprocess/myword2vec.py --model data/nuswide/input/word2vec.model
  -->
  then input a word, the program will list top similar words

  
  convert tag words in record file to word vectors.
  $python models/nuswide/preprocess/myword2vec.py --model data/nuswide/input/word2vec.model --input data/nuswide/input/record-tagwords.dat  --output data/nuswide/input/record-tagvec.dat
  2014-12-22 13:55:08,375 - root - INFO - Converting data/nuswide/input/record-tagwords.dat...
  2014-12-22 13:55:08,375 - gensim.utils - INFO - loading Word2Vec object from data/nuswide/input/word2vec.model
  2014-12-22 13:55:08,858 - gensim.utils - INFO - setting ignored attribute syn0norm to None
  2014-12-22 13:55:34,239 - root - INFO - Processed 203598 lines

  convert label word into label vector:
  $python models/nuswide/preprocess/myword2vec.py --model data/nuswide/input/word2vec.model --input data/nuswide/input/label-dict.txt --output data/nuswide/input/labelvec.txt


1.3 insert records into lmdb (train-lmdb, val-lmdb, test-lmdb)
  create training lmdb with 150000 records
  $./build/tools/convert_nuswide --start=0 --size=150000 data/nuswide/raw_input/images/ data/nuswide/input/record-tagvec.dat data/nuswide/multilabel/train-lmdb
  you should see:
  E1222 14:12:30.209822 19594 convert_nuswide.cpp:234] Processed 1000 files.
  E1222 14:12:36.951107 19594 convert_nuswide.cpp:234] Processed 2000 files.
  ...
  E1222 14:38:10.529342 19594 convert_nuswide.cpp:234] Processed 150000 files.
  E1222 14:38:10.529551 19594 convert_nuswide.cpp:257] Finished
  E1222 14:38:10.529623 19594 convert_nuswide.cpp:258] total lines 150000

  create validation lmdb with 26700 records
  $./build/tools/convert_nuswide --start=150000 --size=26700 data/nuswide/raw_input/images/ data/nuswide/input/record-tagvec.dat data/nuswide/multilabel/val-lmdb
  you should see:
  E1222 14:12:30.209822 19594 convert_nuswide.cpp:234] Processed 1000 files.
  E1222 14:12:36.951107 19594 convert_nuswide.cpp:234] Processed 2000 files.
  E1222 16:08:08.732858 21085 convert_nuswide.cpp:255] Processed 26700 files.
  ...
  E1222 16:08:08.732869 21085 convert_nuswide.cpp:257] Finished
  E1222 16:08:08.732878 21085 convert_nuswide.cpp:258] total lines 176700


  create test lmdb with 26700 records
  $./build/tools/convert_nuswide --start=176700 --size=26700 data/nuswide/raw_input/images/ data/nuswide/input/record-tagvec.dat data/nuswide/multilabel/test-lmdb
  you should see:
  E1222 14:12:30.209822 19594 convert_nuswide.cpp:234] Processed 1000 files.
  E1222 14:12:36.951107 19594 convert_nuswide.cpp:234] Processed 2000 files.
  ...
  E1222 16:58:59.805328 24574 convert_nuswide.cpp:255] Processed 26700 files.
  E1222 16:58:59.805338 24574 convert_nuswide.cpp:257] Finished
  E1222 16:58:59.805347 24574 convert_nuswide.cpp:258] total lines 203400

  compute image mean using training data:
  $./build/tools/compute_image_mean data/nuswide/input/multilabel/train-lmdb data/nuswide/multilabel/image_mean.binaryproto
  you should see:
  E1222 17:03:15.526893 24803 compute_image_mean.cpp:140] Processed 10000 files.
  ...
  E1222 17:04:08.796958 24803 compute_image_mean.cpp:140] Processed 150000 files.



2. MultiLabel Experiment
2.1 Supervised DCNN+NLM
2.1.1 train image path
  ./models/nuswide/multilabel/train_image.sh
2.1.2 train text path
  ./models/nuswide/multilabel/train_text.sh
2.1.3 train joint model
  ./models/nuswide/multilabel/train_joint.sh

2.2 Devise-Label
  ./models/nuswide/multilabel/train_devise.sh

2.3 Devise-Tag
  ./models/nuswide/multilabel/train_devise_tag.sh

3. SingleLabel Experiment
3.1.1 train image path
  ./models/nuswide/singlelabel/train_image.sh
3.1.2 train text path
  ./models/nuswide/singlelabel/train_text.sh
3.1.3 train joint model
  ./models/nuswide/singlelabel/train_joint.sh

3.2 Devise-Label
  ./models/nuswide/singlelabel/train_devise.sh

3.3 Devise-Tag
  ./models/nuswide/singlelabel/train_devise_tag.sh
