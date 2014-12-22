#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
template <typename Dtype>
void RankHingeLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  margin_ = this->layer_param_.rank_hinge_param().margin();
}


template <typename Dtype>
void RankHingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* dptr = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* mutable_label = bottom[1]->mutable_cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  for (int i=0;i<num;i++){
    Dtype nrm;
    caffe_nrm2(dim, label+i*dim,&nrm); 
    caffe_axpy(dim, Dtype(1)/nrm, label+i*dim, mutable_label+i*dim);
  }
  Dtype loss=Dtype(0);
  for (int i = 0; i < num; ++i) {
    int irrelevant_idx;
    do{
      irrelevant_idx=caffe_rng_rand()%num;
    }while(irrelevant_idx==i);
    caffe_sub(dim, mutable_label+irrelevant_idx*dim, mutable_label+i*dim, bottom_diff+i*dim);
    Dtype tmp = caffe_cpu_dot(dim, dptr+i*dim, bottom_diff+i*dim);
    if(margin_+tmp>0)
      loss+=margin_+tmp;
    else
      caffe_set(dim, Dtype(0), bottom_diff+i*dim);
  }
  
  *(*top)[0]->mutable_cpu_data()=loss;
}

template <typename Dtype>
void RankHingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (loss_weight!=Dtype(1))
      caffe_axpy((*bottom)[0]->count(), 
                loss_weight, 
                (*bottom)[0]->cpu_diff(),
                (*bottom)[0]->mutable_cpu_diff());
  }
}

INSTANTIATE_CLASS(RankHingeLossLayer);

}  // namespace caffe
