#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DotLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  nrma_.Reshape(bottom[0]->num(), 1, 1, 1);
  nrmb_.Reshape(bottom[0]->num(), 1, 1, 1);
  if(bottom.size()>2)
    CHECK_EQ(bottom[2]->count(),bottom[2]->num());
  if(bottom.size()>3)
    CHECK_EQ(bottom[3]->count(),bottom[3]->num());
}


template <typename Dtype>
void DotLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype* nrma=nrma_.mutable_cpu_data();
  Dtype* nrmb=nrmb_.mutable_cpu_data();
  const Dtype* dptra=bottom[0]->cpu_data();
  const Dtype* dptrb=bottom[1]->cpu_data();
  Dtype loss =Dtype(0);
  for(int i=0, dim = bottom[0]->count()/bottom[0]->num();i<bottom[0]->num();i++){
    caffe_nrm2(dim, dptra+i*dim,nrma+i); 
    caffe_nrm2(dim, dptrb+i*dim, nrmb+i); 
    loss+=caffe_cpu_dot(dim, dptra+i*dim, dptrb+i*dim)/(nrma[i]*nrmb[i]);
  }
  (*top)[0]->mutable_cpu_data()[0] = loss/bottom[0]->num();
}

template <typename Dtype>
void DotLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* loss2=NULL,*loss3=NULL;
  if(bottom->size()>3){
    loss2=(*bottom)[2]->cpu_data();
    loss3=(*bottom)[3]->cpu_data();
  }

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype *nrm=(i==0)?nrmb_.cpu_data():nrma_.cpu_data();
      const Dtype *dptr=(*bottom)[1-i]->cpu_data();
      Dtype *gptr=(*bottom)[i]->mutable_cpu_diff();
      int num=(*bottom)[i]->num();
      int dim=(*bottom)[i]->count()/num;
      for(int j=0;j<num;j++){
        Dtype alpha =Dtype(0);
        if(loss2==NULL||sign*(loss2[j]-loss3[j])>0)
          alpha=-1 * top[0]->cpu_diff()[0]/nrm[j]/num;
        caffe_cpu_axpby(
            dim, 
            alpha,                       
            dptr+j*dim,     
            Dtype(0),                  
            gptr+j*dim);
      }
    }
  }
}

INSTANTIATE_CLASS(DotLossLayer);

}  // namespace caffe
