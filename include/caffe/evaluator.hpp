#ifndef _EVALUATOR_H_
#define _EVALUATOR_H_
#include <cblas.h>
#include <assert.h>
#include <vector>
#include <utility>
#include <cstring>

namespace evaluator {
/**
 * Search metric.
 * Currently only support kCosine.
 */
typedef enum {
  kCosine,
  kEuclidean
} Metric;

/**
 * Evaluate search performance.
 */
template<typename T>
class Searcher{
 /**
  * GTest class to access protected or private members and functions.
  */
 friend class SearcherTest;
 public:
  /**
   * Constructor to init all members to 0 or nullptr.
   */
  Searcher();
  /**
   * Destructor to free all allocated memory.
   */
  ~Searcher();
  /**
   * Constructor to allocate memory and init ground truth matrix, etc.
   * @param num_queries
   * @param num_points number of database points
   * @param label_dim column of the label matrix
   * @param label label matrix for all db points, element [i,j]=k indicates
   * the i-th data point is associated with label k.
   */
  Searcher(int num_queries, int num_points, int label_dim, const T* label);
  /**
   * Search against db points.
   * This function just calc similarity, between each query and db point.
   * if label is provided, the internal ground truth matrix will be re-created.
   * otherwise, it uses the exisitng ground truth matrix. for the second case,
   * the num_points and num_queries should be the same as the existing values.
   * point_dim can be diff to exisitng value. if diff, then reallocate memory
   * for the query points.
   *
   * @param db database points to be searched of shape num_points x point_dim.
   * \copydetails Searcher::Searcher(int, int, int, int, T*);
   * @return similarity matrix for each query and data point
   */
  const T* Search(const T* db, int num_points, int point_dim,
      int num_queries, const T* label,  int label_dim, Metric metric=kCosine);
  /**
   * Search against db points.
   * Assume the internal ground truth has been created
   * @param db  the shape of db is num_points_ x point_dim, num_points_ is set
   * before.
   * @param point_dim dimension of each point
   * @return similarity matrix for each query and data point
   */
  const T* Search(const T* db, int point_dim, Metric metric=kCosine);
  /**
   * Calc MAP of the last search.
   * @param simmat similarity matrix for every query and data point, if null, use
   * the sim matrix from last search.
   * @param topk consider only topk results,0 for all results.
   * @return MAP score of last search
   */
  float GetMAP(const T* simmat, int topk=0);
  /**
   * Cacl precison and recall of the last search.
   * @param n number of values for precision/recall
   * @return precision recall pairs.
   */
  std::vector<std::pair<float, float> > GetPrecisionRecall(int n=11);
  /**
   * Create ground truth matrix and calc relevant points for each query.
   * \copydetails Searcher::Searcher(int,int, int, int, T*);
   */
  void SetupGroundTruth(int num_queries, int num_points,
            int label_dim, const T *label);

  int query_id_size(){
    return query_id_.size();
  }
  int query_id(int k){
    return query_id_[k];
  }

 protected:
  /**
   * Generate query points IDs.
   * @param num_queries num of query IDs to generate.
   * @param num_points total number of db points
   */
  void GenQueryIDs(int num_queries,int num_points);
  /**
   * Create ground truth matrix, where element at position (i,j) is 1 if i-th
   * query and j-th point share at least one same label.
   */
  T* CreateGroundTruthMatrix(const std::vector<int>& query_id, const T* label,
                              int num_points, int label_dim);
  /**
   * Sum each row of a matrix.
   */
  T* SumRow(const T* mat, int nrow, int ncol);

  /**
   * calc L2 norm of ponints by calling cblas.
   */
  T myblas_nrm2(const int N, const T *X, const int incX);
  /**
   * call cblas to find largest element's index.
   */
  CBLAS_INDEX myblas_iamax(const int N, const T  *X, const int incX);
  /**
   * wrap for type 'double' and 'float', call cblas's gemm.
   */
  void myblas_gemm(const enum CBLAS_ORDER Order,
                   const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const T alpha, const T *A,
                   const int lda, const T *B, const int ldb,
                   const T beta, T *C, const int ldc);
  /**
   * wrap for type 'double' and 'float', call cblas's gemv.
   */
  void myblas_gemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const T alpha, const T *A, const int lda,
                 const T *X, const int incX, const T beta,
                 T *Y, const int incY);

 protected:
  //! query ids to extrac query points from db points
  std::vector<int> query_id_;
  //! num of queries
  int num_queries_;
  //! query points
  T * query_;
  //! point dimension
  int point_dim_;
  //! similarity from query points to db points
  T * sim_;
  //! num of points in db
  int num_points_;
  //! ground truth matrix one row for one query point
  T * gndmat_;
  //! num of relevant points to each query point
  T * num_relevant_;
};

}
#endif
