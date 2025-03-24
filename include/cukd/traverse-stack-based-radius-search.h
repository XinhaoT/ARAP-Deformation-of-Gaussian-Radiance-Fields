#pragma once

#include "cukd/helpers.h"

namespace cukd {
    template <int k>
    struct RadiusResult {
        inline __device__ void initialize() { count = 0;}

        inline __device__ void processCandidate(int candPrimID)
        {
            indices[count] = candPrimID;
            count += 1;
            // if (count == k){
            //   printf("ERROR: over the set maximum number of covering samples");
            // }
        }

        int indices[k];
        int count;
    };



  template<typename data_t,
           typename data_traits=default_data_traits<data_t>>
  inline __device__
  void traverse_radius( typename data_traits::point_t queryPoint,
                        const data_t *d_nodes,
                        int numPoints,
                        float radius,
                        int* result,
                        int &count,
                        const int maximum
  )
  {
    using point_t  = typename data_traits::point_t;
    using scalar_t = typename scalar_type_of<point_t>::type;
    enum { num_dims = num_dims_of<point_t>::value };
    
    // result.initialize();
    /* can do at most 2**30 points... */
    struct StackEntry {
      int   nodeID;
      float sqrDist;
    };
    StackEntry stackBase[30];
    StackEntry *stackPtr = stackBase;

    /*! current node in the tree we're traversing */
    int curr = 0;
    count = 0;
    while (true) {
      while (curr < numPoints) {
        const int  curr_dim
          = data_traits::has_explicit_dim
          ? data_traits::get_dim(data_traits::get_point(d_nodes[curr]))
          : (BinaryTree::levelOf(curr) % num_dims);
        CUKD_STATS(if (cukd::g_traversalStats) ::atomicAdd(cukd::g_traversalStats,1));
        const data_t &curr_node  = d_nodes[curr];
        const auto sqrDist = sqrDistance(data_traits::get_point(curr_node),queryPoint);
        
        if (sqrDist <= radius){
          result[count] = curr;
          count += 1;
        }
        if (count >= maximum){
          printf("ERROR! Exceed maximum occurs! maximun is: %d\n", maximum);
          return;
        }

        const auto node_coord   = data_traits::get_coord(data_traits::get_point(curr_node),curr_dim);
        const auto query_coord  = get_coord(queryPoint,curr_dim);
        const bool  leftIsClose = query_coord < node_coord;
        const int   lChild = 2*curr+1;
        const int   rChild = lChild+1;

        const int closeChild = leftIsClose?lChild:rChild;
        const int farChild   = leftIsClose?rChild:lChild;
        
        const float sqrDistToPlane = sqr(query_coord - node_coord);
        if (sqrDistToPlane < radius && farChild < numPoints) {
          stackPtr->nodeID  = farChild;
          stackPtr->sqrDist = sqrDistToPlane;
          ++stackPtr;
        }
        curr = closeChild;
      }

      while (true) {
        if (stackPtr == stackBase) 
          return;
        --stackPtr;
        if (stackPtr->sqrDist >= radius)
          continue;
        curr = stackPtr->nodeID;
        break;
      }
    }
  }

}