#include <glog/logging.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include "caffe/evaluator.hpp"
#include "caffe/util/math_functions.hpp"

namespace evaluator {
template<typename T>
Searcher<T>::~Searcher(){
  if(query_!=NULL)
    delete query_;
  if(sim_!=NULL)
    delete sim_;
  if(gndmat_!=NULL)
    delete gndmat_;
  if(num_relevant_!=NULL)
    delete num_relevant_;
}

template<typename T>
Searcher<T>::Searcher(){
  point_dim_=num_points_=num_queries_=0;
  query_id_=NULL;
  query_=NULL;
  sim_=NULL;
  gndmat_=NULL;
  num_relevant_=NULL;
}

template<typename T>
Searcher<T>::Searcher(int num_queries, int num_points, int label_dim, 
    const T* label){
  assert(num_queries<num_points);
  query_=NULL;
  sim_=NULL;
  gndmat_=NULL;
  num_relevant_=NULL;
  SetupGroundTruth(num_queries, num_points, label_dim, label);
}

template<typename T>
void Searcher<T>::GenQueryIDs(int num_queries, int num_points){
  if(query_id_!=NULL)
    delete query_id_;
  query_id_= new int[num_queries];
  caffe::caffe_rng_int_uniform(num_queries, 0, num_points-1, query_id_);
  num_queries_=num_queries;
}

template<typename T>
void Searcher<T>::SetupGroundTruth(int num_queries, int num_points,
    int label_dim, const T *label){
  num_points_=num_points;
  if(num_queries_!=num_queries_){
    GenQueryIDs(num_queries, num_points);
  }
  num_queries_=num_queries;
  if(gndmat_!=NULL)
    delete gndmat_;
  gndmat_=CreateGroundTruthMatrix(query_id_, label, num_points_, label_dim);
  if(num_relevant_!=NULL)
    delete num_relevant_;
  num_relevant_=SumRow(gndmat_, num_queries_, num_points_);
}
/**
 * label is of shape num_pointer x label_dim, value starts from 0
 */
template<typename T>
T* Searcher<T>::CreateGroundTruthMatrix(const int* query_id,
                                        const T* label,
                                        int num_points,
                                        int label_dim){
  // generate ground truth matrix
  int num_queries=num_queries_;
  int maxlabelidx=myblas_iamax(label_dim*num_points, label, 1);
  // binary label
  int blabel_dim=label[maxlabelidx]+1;
  T* query_label=new T[num_queries*blabel_dim];
  memset(query_label, 0, sizeof(int)*num_queries*blabel_dim);
  for(int i=0;i<num_queries;i++){
    int j=0;
    int lb=label[query_id[i]*label_dim+j];
    while(lb!=-1&&j<label_dim){
      query_label[i*blabel_dim+lb]=1;
      j++;
      lb=label[query_id_[i]*label_dim+j];
    }
  }
  T* db_label=new T[num_points*blabel_dim];
  memset(db_label, 0, sizeof(int)*num_points*blabel_dim);
  for(int i=0;i<num_points;i++){
    int j=0;
    int lb=label[i*label_dim+j];
    while(lb!=-1&&j<label_dim){
      db_label[i*blabel_dim+lb]=1;
      j++;
      lb=label[i*label_dim+j];
    }
  }
  T* gndmat=new T[num_queries*num_points];
  myblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_queries, num_points,
      blabel_dim, 1.0f, query_label, blabel_dim, db_label, blabel_dim,
      0.0f, gndmat, num_points);
  for(int i=0;i<num_points*num_queries;i++)
    gndmat[i]=gndmat[i]>0.0f?1.0f:0.0f;

  delete db_label;
  delete query_label;
  return gndmat;
}

template<typename T>
T* Searcher<T>::SumRow(const T* mat, int nrow, int ncol){
  // compute num relevant points for each query point
  T*  sum=new T[nrow];
  T *one=new T[ncol];
  for(int i=0;i<ncol;i++)
    one[i]=1.0f;
  myblas_gemv(CblasRowMajor, CblasNoTrans, nrow, ncol, 1.0f, mat, ncol,
      one,1, 0.0f, sum, 1);
  delete one;
  return sum;
}

template<typename T>
const T* Searcher<T>::Search(const T* db, int point_dim, Metric metric){
  point_dim_=point_dim;
  if(query_==NULL)
    query_=new T[num_queries_*point_dim_];
  if(sim_==NULL)
    sim_=new T[num_queries_*num_points_];
  // prepare query points
  for(int i=0;i<num_queries_;i++){
    memcpy(query_+i*point_dim_, db+query_id_[i]*point_dim_,
        sizeof(T)*point_dim_);
  }
  if(metric==kCosine){
    // dot query points and db points
    myblas_gemm(CblasRowMajor,CblasNoTrans, CblasTrans, num_queries_,
        num_points_, point_dim_, 1.0f, query_, point_dim_, db, point_dim_,
        0.0f, sim_, num_points_);
    // L2 norm of query points
    T* query_nrm2=new T[num_queries_];
    for(int i=0; i<num_queries_;i++)
      query_nrm2[i]=myblas_nrm2(point_dim_, query_+i*point_dim_, 1);
    // L2 norm of db points
    T* db_nrm2=new T[num_points_];
    for(int i=0;i<num_points_;i++)
      db_nrm2[i]=myblas_nrm2(point_dim_, db+i*point_dim_,1);
    // normalize similarity
    for(int i=0, k=0;i<num_queries_;i++){
      for(int j=0;j<num_points_;j++)
        sim_[k++]/=query_nrm2[i]*db_nrm2[j];
    }
    delete query_nrm2;
    delete db_nrm2;
  }else{
    std::cout<<"ERROR:Not implemented for metric other than cosine";
  }
  return sim_;
}

template<typename T>
const T* Searcher<T>::Search(const T* db, int num_points, int point_dim,
    int num_queries, const T* label, int label_dim, Metric metric){
  if(point_dim!=point_dim_||num_queries!=num_queries_){
    point_dim_=point_dim;
    if(query_!=NULL){
      delete query_;
      query_=NULL;
    }
  }
  if(num_queries!=num_queries_||num_points!=num_points_){
    if(sim_!=NULL){
      delete sim_;
      sim_=NULL;
    }
    num_queries_=num_queries;
    num_points_=num_points;
  }
  if(label!=NULL)
    SetupGroundTruth(num_queries, num_points, label_dim, label);
  return Search(db, point_dim,  metric);
}

template<typename T>
float Searcher<T>::GetMAP(const T* simmat, int topk) {
  assert(gndmat_!=NULL);
  assert(num_relevant_!=NULL);
  const T* mat;
  if(simmat==NULL){
    CHECK(sim_!= NULL);
    mat=sim_;
  }
  else 
    mat=simmat;

  float map=0.0f;
  if(topk==0)
    topk=num_points_;
  for(size_t i=0;i<num_queries_;i++){
    std::vector<std::pair<T, int> > sim;
    for(int j=0;j<num_points_;j++)
      sim.push_back(std::make_pair(mat[i*num_points_+j], j));

    std::partial_sort(sim.begin(), sim.begin()+topk,
        sim.end(), std::greater<std::pair<T, int> >());

    T *gnd=gndmat_+i*num_points_;
    float hits=0.0f, score=0.0f;
    for(int j=0;j<topk;j++){
      if(gnd[sim[j].second]>0){
        hits+=1;
        score+=hits/(1.0+j);
      }
    }
    CHECK_LE(hits,num_relevant_[i]);
    map+=score/std::min(topk*1.0f, static_cast<float>(num_relevant_[i]));
  }
  return map/num_queries_;
}
template<>
void Searcher<float>::myblas_gemm(const enum CBLAS_ORDER Order,
                   const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const float alpha, const float *A,
                   const int lda, const float *B, const int ldb,
                   const float beta, float *C, const int ldc){
  cblas_sgemm(Order, TransA, TransB, M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
}
template<>
void Searcher<double>::myblas_gemm(const enum CBLAS_ORDER Order,
                   const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const double alpha, const double *A,
                   const int lda, const double *B, const int ldb,
                   const double beta, double *C, const int ldc){
  cblas_dgemm(Order, TransA, TransB, M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
}
template<>
void Searcher<float>::myblas_gemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY){
  cblas_sgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}
template<>
void Searcher<double>::myblas_gemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY){
  cblas_dgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}
template<>
CBLAS_INDEX Searcher<float>::myblas_iamax(const int N, const float  *X,
    const int incX){
  return cblas_isamax(N,X,incX);
}
template<>
CBLAS_INDEX Searcher<double>::myblas_iamax(const int N, const double  *X,
    const int incX){
  return cblas_idamax(N,X,incX);
}
template<>
float Searcher<float>::myblas_nrm2(const int N, const float *X, const int incX){
  return cblas_snrm2(N,X,incX);
}
template<>
double Searcher<double>::myblas_nrm2(const int N, const double *X, const int incX){
  return cblas_dnrm2(N,X,incX);
}

template class Searcher<float>;
template class Searcher<double>;
} /* Evaluator */
