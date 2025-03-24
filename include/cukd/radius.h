#pragma once

#include "cukd/common.h"
#include "cukd/helpers.h"
#include "cukd/data.h"
#include "cukd/spatial-kdtree.h"
#include "cukd/point_struct.h"
#include "traverse-stack-based-radius-search.h"

namespace cukd {
  struct PointPlusPayload_traits
    :public default_data_traits<float3>
  {
    using point_t = float3;
    static inline __both__
    float3 const &get_point(const PointPlusPayload &data) {return data.position;}
  };
  
    namespace stackBased {
    /*! default, stack-based find-closest point kernel, with simple
      point-to-plane-distance test for culling subtrees 
      
      \returns the ID of the point that's closest to the query point,
      or -1 if none could be found within the given serach radius
    */
    template<
      /*! type of data point(s) that the tree is built over (e.g., float3) */
      typename data_t, // The maximum number of containing samples
      /*! traits that describe these points (float3 etc have working defaults */
      typename data_traits=default_data_traits<data_t>>
    inline __device__
    void radiusSearchCUDA(typename data_traits::point_t queryPoint,
            const data_t *dataPoints,
            int numDataPoints,
            float radius,
            int* result,
            int &count,
            const int maximum);
  }
}

// ==================================================================
// IMPLEMENTATION SECTION
// ==================================================================

namespace cukd {
    template<
      typename data_t,
      typename data_traits=default_data_traits<data_t>>
    inline __device__
    void stackBased::radiusSearchCUDA(typename data_traits::point_t queryPoint,
            const data_t *dataPoints,
            int numDataPoints,
            float radius,
            int* result,
            int &count,
            const int maximum)
    {
        // printf("begin radius searching, the maximun containing is set to %d \n", k);
        traverse_radius<data_t, data_traits>(queryPoint, dataPoints, numDataPoints, radius, result, count, maximum);
        return;
    }
}
