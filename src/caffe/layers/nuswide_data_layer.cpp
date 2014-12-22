#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void NuswideDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  CHECK_GE(top->size(),2)<<"Nuswide Layer has at least 2 top blobs";
  // Initialize DB
  switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      {
        leveldb::DB* db_temp;
        leveldb::Options options = GetLevelDBOptions();
        options.create_if_missing = false;
        LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
        leveldb::Status status = leveldb::DB::Open(
            options, this->layer_param_.data_param().source(), &db_temp);
        CHECK(status.ok()) << "Failed to open leveldb "
          << this->layer_param_.data_param().source() << std::endl
          << status.ToString();
        this->db_.reset(db_temp);
        this->iter_.reset(this->db_->NewIterator(leveldb::ReadOptions()));
        this->iter_->SeekToFirst();
      }
      break;
    case DataParameter_DB_LMDB:
      CHECK_EQ(mdb_env_create(&this->mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
      CHECK_EQ(mdb_env_set_mapsize(this->mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
      CHECK_EQ(mdb_env_open(this->mdb_env_,
            this->layer_param_.data_param().source().c_str(),
            MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
      CHECK_EQ(mdb_txn_begin(this->mdb_env_, NULL, MDB_RDONLY, &this->mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
      CHECK_EQ(mdb_open(this->mdb_txn_, NULL, 0, &this->mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
      CHECK_EQ(mdb_cursor_open(this->mdb_txn_, this->mdb_dbi_, &this->mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
      LOG(INFO) << "Opening lmdb " << this->layer_param_.data_param().source();
      CHECK_EQ(mdb_cursor_get(this->mdb_cursor_, &this->mdb_key_, &this->mdb_value_, MDB_FIRST),
          MDB_SUCCESS) << "mdb_cursor_get failed";
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
  }

  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      datum.ParseFromString(this->iter_->value().ToString());
      break;
    case DataParameter_DB_LMDB:
      datum.ParseFromArray(this->mdb_value_.mv_data, this->mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
  }

  // image
  int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), crop_size, crop_size);
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.channels(), datum.height(), datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
    << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
    << (*top)[0]->width()<<". Text vector dim: "<<datum.text_size();
  // label
  if (this->output_labels_) {
    //allocate one more for marker
    (*top)[1]->Reshape(this->layer_param_.data_param().batch_size(),
        this->layer_param_.data_param().max_labels()+1, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.data_param().batch_size(),
        this->layer_param_.data_param().max_labels()+1,  1, 1);
  }
  // text
  if (datum.text_size()){
    (*top)[2]->Reshape(this->layer_param_.data_param().batch_size(),
        datum.text_size(), 1, 1);
    this->prefetch_text_.Reshape(this->layer_param_.data_param().batch_size(),
        datum.text_size(), 1, 1);
  }
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
  this->text_dim_=datum.text_size();

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()||this->layer_param_.data_param().skip()) {
    unsigned int skip =0;
    if(this->layer_param_.data_param().skip())
      skip=this->layer_param_.data_param().skip();
    else
      skip= caffe_rng_rand() % this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    if(skip>0&&this->layer_param_.data_param().backend()==DataParameter_DB_LMDB)
      CHECK_EQ(mdb_cursor_get(this->mdb_cursor_, &this->mdb_key_, &this->mdb_value_,
            MDB_FIRST), MDB_SUCCESS);
    while (skip-- > 0) {
      switch (this->layer_param_.data_param().backend()) {
        case DataParameter_DB_LEVELDB:
          this->iter_->Next();
          if (!this->iter_->Valid()) {
            this->iter_->SeekToFirst();
          }
          break;
        case DataParameter_DB_LMDB:
          CHECK_EQ (mdb_cursor_get(this->mdb_cursor_, &this->mdb_key_, &this->mdb_value_, MDB_NEXT)
              ,MDB_SUCCESS);
          break;
        default:
          LOG(FATAL) << "Unknown database backend";
      }
    }
  }
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void NuswideDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_text = this->prefetch_text_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    switch (this->layer_param_.data_param().backend()) {
      case DataParameter_DB_LEVELDB:
        CHECK(this->iter_);
        CHECK(this->iter_->Valid());
        datum.ParseFromString(this->iter_->value().ToString());
        break;
      case DataParameter_DB_LMDB:
        CHECK_EQ(mdb_cursor_get(this->mdb_cursor_, &this->mdb_key_,
              &this->mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
        datum.ParseFromArray(this->mdb_value_.mv_data,
            this->mdb_value_.mv_size);
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
    }

    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);

    if (this->output_labels_) {
      int label_dim=this->layer_param_.data_param().max_labels()+1;
      CHECK_LE(datum.multi_label_size(), label_dim);
      int i=0;
      for(;i<datum.multi_label_size();i++)
        top_label[item_id*label_dim+i]=static_cast<Dtype>(datum.multi_label(i));
      // add a marker
      top_label[item_id*label_dim+i]=-1;
    }

    if (datum.text_size()){
      int m=datum.text_size();
      for(int i=0;i<m;i++)
        top_text[item_id*m+i]=static_cast<Dtype>(datum.text(i));
    }

    // go to the next iter
    switch (this->layer_param_.data_param().backend()) {
      case DataParameter_DB_LEVELDB:
        this->iter_->Next();
        if (!this->iter_->Valid()) {
          // We have reached the end. Restart from the first.
          DLOG(INFO) << "Restarting data prefetching from start.";
          this->iter_->SeekToFirst();
        }
        break;
      case DataParameter_DB_LMDB:
        if (mdb_cursor_get(this->mdb_cursor_, &this->mdb_key_,
              &this->mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
          // We have reached the end. Restart from the first.
          LOG(INFO) << "Restarting data prefetching from start.";
          CHECK_EQ(mdb_cursor_get(this->mdb_cursor_, &this->mdb_key_, &this->mdb_value_,
                MDB_FIRST), MDB_SUCCESS);
          int skip=0;
          if(this->layer_param_.data_param().skip()) 
            skip=this->layer_param_.data_param().skip();
          CHECK(item_id==batch_size-1||skip==0)<<"item_id "<<item_id<<" "<<skip;
          while (skip-- > 0) {
            CHECK_EQ(mdb_cursor_get(this->mdb_cursor_, &this->mdb_key_, &this->mdb_value_, MDB_NEXT)
                ,MDB_SUCCESS);
          }
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
    }
  }
}

INSTANTIATE_CLASS(NuswideDataLayer);

}  // namespace caffe
