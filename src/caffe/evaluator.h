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
   * @param label label matrix for all db points
   */
  explicit Searcher(int num_queries, int num_points,
                    int label_dim, T* label);
  /**
   * Search against db points.
   * This function just calc distances, between each query and db point.
   * \copydetails Searcher::Searcher(int, int, int, int, T*);
   */
  void Search(T* db, int num_points, int point_dim,
      int num_queries, T* label,  int label_dim, Metric metric=kCosine);
  /**
   * \copybrief Search(T*, int, int, int, T*, int, Metric);
   */
  void Search(T* db, int point_dim, Metric metric=kCosine);
  /**
   * Calc MAP of the last search.
   * @param topk consider only topk results,0 for all results.
   * @return MAP score of last search
   */
  float GetMAP(int topk=0);
  /**
   * Cacl precison and recall of the last search.
   * @param n number of values for precision/recall
   * @return precision recall pairs.
   */
  std::vector<std::pair<float, float> > GetPrecisionRecall(int n=11);

 protected:
  /**
   * Generate query points IDs.
   * @param num_queries num of query IDs to generate.
   * @param num_points total number of db points
   */
  void GenQueryIDs(int num_queries,int num_points);
  /**
   * Create ground truth matrix and calc relevant points for each query.
   * \copydetails Searcher::Searcher(int,int, int, int, T*);
   */
  void SetupGroundTruth(int num_queries, int num_points,
            int label_dim, T *label);
  /**
   * Create ground truth matrix, where element at position (i,j) is 1 if i-th
   * query and j-th point share at least one same label.
   */
  T* CreateGroundTruthMatrix(const std::vector<int>& query_id, T* label,
                              int num_points, int label_dim);
  /**
   * Sum each row of a matrix.
   */
  T* SumRow(T* mat, int nrow, int ncol);

  T myblas_nrm2(const int N, const T *X, const int incX);
  CBLAS_INDEX myblas_iamax(const int N, const T  *X, const int incX);
  void myblas_gemm(const enum CBLAS_ORDER Order,
                   const enum CBLAS_TRANSPOSE TransA,
                   const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                   const int K, const T alpha, const T *A,
                   const int lda, const T *B, const int ldb,
                   const T beta, T *C, const int ldc);
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
  //! distance from query points to db points
  T * distance_;
  //! num of points in db
  int num_points_;
  //! ground truth matrix one row for one query point
  T * gndmat_;
  //! num of relevant points to each query point
  T * num_relevant_;
};

}
#endif
