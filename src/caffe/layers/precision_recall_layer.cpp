#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PrecisionRecallLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  top_k_ = this->layer_param_.pr_param().top_k();
}

template <typename Dtype>
void PrecisionRecallLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
      << "top_k must be less than or equal to the number of classes.";
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  // one for precision, one for recall
  (*top)[0]->Reshape(2, 1, 1, 1);
}

template <typename Dtype>
void PrecisionRecallLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  int label_dim=bottom[1]->count()/num;
  vector<Dtype> maxval(top_k_+1);
  Dtype recall=0.f, precision=0.f;
  for (int i = 0; i < num; ++i) {
    // Top-k accuracy
    std::vector<std::pair<Dtype, int> > bottom_data_vector;
    for (int j = 0; j < dim; ++j) {
      bottom_data_vector.push_back(
          std::make_pair(bottom_data[i * dim + j], j));
    }
    std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
        bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
    // sort label id in ascending order
    // std::sort(bottom_data_vector.begin(), bottom_data_vector.begin()+top_k_);
    Dtype ncorrect = 0, ntotal=0;
    // check if true label is in top k predictions
    for(int j=0;bottom_label[label_dim*i+j]!=-1&&j<label_dim;j++){
      int label=static_cast<int>(bottom_label[label_dim*i+j]);
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label){
          ++ncorrect;
          break;
        }
      }
      ++ntotal;
    }
    CHECK_GT(ntotal,0);
    precision+=ncorrect/top_k_;
    recall+=ncorrect/ntotal;
  }

  // precision
  (*top)[0]->mutable_cpu_data()[0] = precision / num;
  // recall
  (*top)[0]->mutable_cpu_data()[1] = recall / num;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(PrecisionRecallLayer);

}  // namespace caffe
