#ifndef _EVALUATOR_H_
#define _EVALUATOR_H_
#include <cblas.h>
#include <assert.h>
#include <vector>
#include <utility>
#include <chrono>
#include <random>
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
   * @param point_dim dimension of each point
   * @param label_dim column of the label matrix
   * @param label label matrix for all db points
   */
  explicit Searcher(int num_queries, int num_points,
                    int point_dim, int label_dim, float* label);
  /**
   * Search against db points.
   * This function just calc distances, between each query and db point.
   * \copydetails Searcher::Searcher(int, int, int, int, float*);
   */
  void Search(float* db, int num_points, int point_dim,
      int num_queries, float* label,  int label_dim, Metric metric=kCosine);
  /**
   * \copybrief Search(float*, int, int, int, float*, int, Metric);
   */
  void Search(float* db, Metric metric=kCosine);
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
   * Allocate memory and create ground truth matrix.
   * \copydetails Searcher::Searcher(int,int, int, int, float*);
   */
  void Setup(int num_queries, int num_points,
            int point_dim, int label_dim,float *label);
  /**
   * Create ground truth matrix, where element at position (i,j) is 1 if i-th
   * query and j-th point share at least one same label.
   */
  float* CreateGroundTruthMatrix(const std::vector<int>& query_id, float* label,
                              int num_points, int label_dim);
  /**
   * Sum each row of a matrix.
   */
  float* SumRow(float* mat, int nrow, int ncol);

 protected:
  //! query ids to extrac query points from db points
  std::vector<int> query_id_;
  //! num of queries
  int num_queries_;
  //! query points
  float * query_;
  //! point dimension
  int point_dim_;
  //! distance from query points to db points
  float * distance_;
  //! num of points in db
  int num_points_;
  //! ground truth matrix one row for one query point
  float * gndmat_;
  //! num of relevant points to each query point
  float * num_relevant_;
};

}
#endif
