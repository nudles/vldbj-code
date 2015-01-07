#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>  // NOLINT(readability/streams)

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
template <typename Dtype>
void RankHingeLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom.size(),3);
  margin_ = this->layer_param_.rank_hinge_param().margin();
  if(this->layer_param_.rank_hinge_param().has_label_dict()){
    string label_dict_fname=this->layer_param_.rank_hinge_param().label_dict();
    std::ifstream fin(this->layer_param_.rank_hinge_param().label_dict());
    std::vector<string> lines;
    while(!fin.eof()){
      string line;
      std::getline(fin, line);
      if(fin.eof())
        break;
      lines.push_back(line);
    }
    int dim=bottom[0]->channels();
    label_vec_.Reshape(lines.size(), dim, 1, 1);
    for(int i=0;i<lines.size();i++){
      std::vector<std::string> strs;
      boost::split(strs, lines[i], boost::is_any_of(" "));
      int labelid=boost::lexical_cast<int>(strs[0]);
      CHECK_EQ(dim, strs.size()-1);
      Dtype *vec=label_vec_.mutable_cpu_data()+dim*labelid;
      for(int i=0;i<dim;i++)
        vec[i]=boost::lexical_cast<Dtype>(strs[i+1]);
    }
  }
}


template <typename Dtype>
void RankHingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* dptr = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label;
  Dtype* mutable_label ;
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim =bottom[0]->channels(); 
  if(label_vec_.count()>0){
    mutable_label=new Dtype[count];
    label=mutable_label;
    for(int i=0, labeldim=bottom[1]->channels();i<num;i++){
      for(int j=0;j<labeldim;j++){
        int labelid=static_cast<int>(bottom[1]->cpu_data()[i*labeldim+j]);
        if(labelid==-1) break;
        CHECK_LT(labelid, label_vec_.num());
        caffe_cpu_axpby(dim, Dtype(1), label_vec_.cpu_data()+labelid*dim, 
            j==0?Dtype(0):Dtype(1), mutable_label+dim*i);
      }
    }
  }else{
    label=bottom[2]->cpu_data();
    mutable_label=bottom[2]->mutable_cpu_data();
  }
  for (int i=0;i<num;i++){
    Dtype nrm;
    caffe_nrm2(dim, label+i*dim,&nrm); 
    caffe_axpy(dim, Dtype(1)/nrm, label+i*dim, mutable_label+i*dim);
  }
  Dtype loss=Dtype(0);
  for (int i = 0; i < num; ++i) {
    int irrelevant_idx, tmpcount=0;
    while(true){
      tmpcount++;
      irrelevant_idx=caffe_rng_rand()%num;
      if(label_vec_.channels()==2){
        if(bottom[1]->cpu_data()[2*irrelevant_idx]!=bottom[1]->cpu_data()[2*i]||tmpcount>10) 
          break;
      }else if(irrelevant_idx!=i)
        break;
    }
    caffe_sub(dim, mutable_label+irrelevant_idx*dim, mutable_label+i*dim, bottom_diff+i*dim);
    Dtype tmp = caffe_cpu_dot(dim, dptr+i*dim, bottom_diff+i*dim);
    if(margin_+tmp>0)
      loss+=margin_+tmp;
    else
      caffe_set(dim, Dtype(0), bottom_diff+i*dim);
  }
  if(label_vec_.count()>0)
    delete label;
  *(*top)[0]->mutable_cpu_data()=loss/num;
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
