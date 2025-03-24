#include "cukd/builder.h"
#include "cukd/knn.h"
#include "cudakdtree.cuh"
#include "cukd/radius.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
namespace cg = cooperative_groups;

__device__ void Construct_SH_Rotation_MatrixCUDA(Eigen::Matrix<float, 3, 3>& sh1, 
Eigen::Matrix<float, 5, 5>& sh2,
Eigen::Matrix<float, 7, 7>& sh3,
Eigen::Matrix<float, 3, 3> rots ){
    sh1.row(0)(0) = rots.row(1)(1);
    sh1.row(0)(1) = rots.row(1)(2);
    sh1.row(0)(2) = rots.row(1)(0);
    sh1.row(1)(0) = rots.row(2)(1);
    sh1.row(1)(1) = rots.row(2)(2);
    sh1.row(1)(2) = rots.row(2)(0);
    sh1.row(2)(0) = rots.row(0)(1);
    sh1.row(2)(1) = rots.row(0)(2);
    sh1.row(2)(2) = rots.row(0)(0);

    sh2.row(0)(0) = sqrt( 1.0 /  4.0) * ((sh1.row(2)(2) * sh1.row(0)(0) + sh1.row(2)(0) * sh1.row(0)(2)) + (sh1.row(0)(2) * sh1.row(2)(0) + sh1.row(0)(0) * sh1.row(2)(2)));
    sh2.row(0)(1) = (sh1.row(2)(1) * sh1.row(0)(0) + sh1.row(0)(1) * sh1.row(2)(0));
    sh2.row(0)(2) = sqrt( 3.0 /  4.0) * (sh1.row(2)(1) * sh1.row(0)(1) + sh1.row(0)(1) * sh1.row(2)(1));
    sh2.row(0)(3) = (sh1.row(2)(1) * sh1.row(0)(2) + sh1.row(0)(1) * sh1.row(2)(2));
    sh2.row(0)(4) = sqrt( 1.0 /  4.0) * ((sh1.row(2)(2) * sh1.row(0)(2) - sh1.row(2)(0) * sh1.row(0)(0)) + (sh1.row(0)(2) * sh1.row(2)(2) - sh1.row(0)(0) * sh1.row(2)(0)));

    sh2.row(1)(0) = sqrt( 1.0 /  4.0) * ((sh1.row(1)(2) * sh1.row(0)(0) + sh1.row(1)(0) * sh1.row(0)(2)) + (sh1.row(0)(2) * sh1.row(1)(0) + sh1.row(0)(0) * sh1.row(1)(2)));
    sh2.row(1)(1) = sh1.row(1)(1) * sh1.row(0)(0) + sh1.row(0)(1) * sh1.row(1)(0);
    sh2.row(1)(2) = sqrt( 3.0 /  4.0) * (sh1.row(1)(1) * sh1.row(0)(1) + sh1.row(0)(1) * sh1.row(1)(1));
    sh2.row(1)(3) = sh1.row(1)(1) * sh1.row(0)(2) + sh1.row(0)(1) * sh1.row(1)(2);
    sh2.row(1)(4) = sqrt( 1.0 /  4.0) * ((sh1.row(1)(2) * sh1.row(0)(2) - sh1.row(1)(0) * sh1.row(0)(0)) + (sh1.row(0)(2) * sh1.row(1)(2) - sh1.row(0)(0) * sh1.row(1)(0)));

    sh2.row(2)(0) = sqrt( 1.0 /  3.0) * (sh1.row(1)(2) * sh1.row(1)(0) + sh1.row(1)(0) * sh1.row(1)(2)) + -sqrt( 1.0 /  12.0) * ((sh1.row(2)(2) * sh1.row(2)(0) + sh1.row(2)(0) * sh1.row(2)(2)) + (sh1.row(0)(2) * sh1.row(0)(0) + sh1.row(0)(0) * sh1.row(0)(2)));
    sh2.row(2)(1) = sqrt( 4.0 /  3.0) * sh1.row(1)(1) * sh1.row(1)(0) + -sqrt( 1.0 /  3.0) * (sh1.row(2)(1) * sh1.row(2)(0) + sh1.row(0)(1) * sh1.row(0)(0));
    sh2.row(2)(2) = sh1.row(1)(1) * sh1.row(1)(1) + -sqrt( 1.0 /  4.0) * (sh1.row(2)(1) * sh1.row(2)(1) + sh1.row(0)(1) * sh1.row(0)(1));
    sh2.row(2)(3) = sqrt( 4.0 /  3.0) * sh1.row(1)(1) * sh1.row(1)(2) + -sqrt( 1.0 /  3.0) * (sh1.row(2)(1) * sh1.row(2)(2) + sh1.row(0)(1) * sh1.row(0)(2));
    sh2.row(2)(4) = sqrt( 1.0 /  3.0) * (sh1.row(1)(2) * sh1.row(1)(2) - sh1.row(1)(0) * sh1.row(1)(0)) + -sqrt( 1.0 /  12.0) * ((sh1.row(2)(2) * sh1.row(2)(2) - sh1.row(2)(0) * sh1.row(2)(0)) + (sh1.row(0)(2) * sh1.row(0)(2) - sh1.row(0)(0) * sh1.row(0)(0)));

    sh2.row(3)(0) = sqrt( 1.0 /  4.0) * ((sh1.row(1)(2) * sh1.row(2)(0) + sh1.row(1)(0) * sh1.row(2)(2)) + (sh1.row(2)(2) * sh1.row(1)(0) + sh1.row(2)(0) * sh1.row(1)(2)));
    sh2.row(3)(1) = sh1.row(1)(1) * sh1.row(2)(0) + sh1.row(2)(1) * sh1.row(1)(0);
    sh2.row(3)(2) = sqrt( 3.0 /  4.0) * (sh1.row(1)(1) * sh1.row(2)(1) + sh1.row(2)(1) * sh1.row(1)(1));
    sh2.row(3)(3) = sh1.row(1)(1) * sh1.row(2)(2) + sh1.row(2)(1) * sh1.row(1)(2);
    sh2.row(3)(4) = sqrt( 1.0 /  4.0) * ((sh1.row(1)(2) * sh1.row(2)(2) - sh1.row(1)(0) * sh1.row(2)(0)) + (sh1.row(2)(2) * sh1.row(1)(2) - sh1.row(2)(0) * sh1.row(1)(0)));

    sh2.row(4)(0) = sqrt( 1.0 /  4.0) * ((sh1.row(2)(2) * sh1.row(2)(0) + sh1.row(2)(0) * sh1.row(2)(2)) - (sh1.row(0)(2) * sh1.row(0)(0) + sh1.row(0)(0) * sh1.row(0)(2)));
    sh2.row(4)(1) = (sh1.row(2)(1) * sh1.row(2)(0) - sh1.row(0)(1) * sh1.row(0)(0));
    sh2.row(4)(2) = sqrt( 3.0 /  4.0) * (sh1.row(2)(1) * sh1.row(2)(1) - sh1.row(0)(1) * sh1.row(0)(1));
    sh2.row(4)(3) = (sh1.row(2)(1) * sh1.row(2)(2) - sh1.row(0)(1) * sh1.row(0)(2));
    sh2.row(4)(4) = sqrt( 1.0 /  4.0) * ((sh1.row(2)(2) * sh1.row(2)(2) - sh1.row(2)(0) * sh1.row(2)(0)) - (sh1.row(0)(2) * sh1.row(0)(2) - sh1.row(0)(0) * sh1.row(0)(0)));

    sh3.row(0)(0) = sqrt( 1.0 /  4.0) * ((sh1.row(2)(2) * sh2.row(0)(0) + sh1.row(2)(0) * sh2.row(0)(4)) + (sh1.row(0)(2) * sh2.row(4)(0) + sh1.row(0)(0) * sh2.row(4)(4)));
    sh3.row(0)(1) = sqrt( 3.0 /  2.0) * (sh1.row(2)(1) * sh2.row(0)(0) + sh1.row(0)(1) * sh2.row(4)(0));
    sh3.row(0)(2) = sqrt(15.0 / 16.0) * (sh1.row(2)(1) * sh2.row(0)(1) + sh1.row(0)(1) * sh2.row(4)(1));
    sh3.row(0)(3) = sqrt( 5.0 /  6.0) * (sh1.row(2)(1) * sh2.row(0)(2) + sh1.row(0)(1) * sh2.row(4)(2));
    sh3.row(0)(4) = sqrt(15.0 / 16.0) * (sh1.row(2)(1) * sh2.row(0)(3) + sh1.row(0)(1) * sh2.row(4)(3));
    sh3.row(0)(5) = sqrt( 3.0 /  2.0) * (sh1.row(2)(1) * sh2.row(0)(4) + sh1.row(0)(1) * sh2.row(4)(4));
    sh3.row(0)(6) = sqrt( 1.0 /  4.0) * ((sh1.row(2)(2) * sh2.row(0)(4) - sh1.row(2)(0) * sh2.row(0)(0)) + (sh1.row(0)(2) * sh2.row(4)(4) - sh1.row(0)(0) * sh2.row(4)(0)));

    sh3.row(1)(0) = sqrt( 1.0 /  6.0) * (sh1.row(1)(2) * sh2.row(0)(0) + sh1.row(1)(0) * sh2.row(0)(4)) + sqrt( 1.0 /  6.0) * ((sh1.row(2)(2) * sh2.row(1)(0) + sh1.row(2)(0) * sh2.row(1)(4)) + (sh1.row(0)(2) * sh2.row(3)(0) + sh1.row(0)(0) * sh2.row(3)(4)));
    sh3.row(1)(1) = sh1.row(1)(1) * sh2.row(0)(0) + (sh1.row(2)(1) * sh2.row(1)(0) + sh1.row(0)(1) * sh2.row(3)(0));
    sh3.row(1)(2) = sqrt( 5.0 /  8.0) * sh1.row(1)(1) * sh2.row(0)(1) + sqrt( 5.0 /  8.0) * (sh1.row(2)(1) * sh2.row(1)(1) + sh1.row(0)(1) * sh2.row(3)(1));
    sh3.row(1)(3) = sqrt( 5.0 /  9.0) * sh1.row(1)(1) * sh2.row(0)(2) + sqrt( 5.0 /  9.0) * (sh1.row(2)(1) * sh2.row(1)(2) + sh1.row(0)(1) * sh2.row(3)(2));
    sh3.row(1)(4) = sqrt( 5.0 /  8.0) * sh1.row(1)(1) * sh2.row(0)(3) + sqrt( 5.0 /  8.0) * (sh1.row(2)(1) * sh2.row(1)(3) + sh1.row(0)(1) * sh2.row(3)(3));
    sh3.row(1)(5) = sh1.row(1)(1) * sh2.row(0)(4) + (sh1.row(2)(1) * sh2.row(1)(4) + sh1.row(0)(1) * sh2.row(3)(4));
    sh3.row(1)(6) = sqrt( 1.0 /  6.0) * (sh1.row(1)(2) * sh2.row(0)(4) - sh1.row(1)(0) * sh2.row(0)(0)) + sqrt( 1.0 /  6.0) * ((sh1.row(2)(2) * sh2.row(1)(4) - sh1.row(2)(0) * sh2.row(1)(0)) + (sh1.row(0)(2) * sh2.row(3)(4) - sh1.row(0)(0) * sh2.row(3)(0)));

    sh3.row(2)(0) = sqrt( 4.0 / 15.0) * (sh1.row(1)(2) * sh2.row(1)(0) + sh1.row(1)(0) * sh2.row(1)(4)) + sqrt( 1.0 /  5.0) * (sh1.row(0)(2) * sh2.row(2)(0) + sh1.row(0)(0) * sh2.row(2)(4)) + -sqrt(1.0 / 60.0) * ((sh1.row(2)(2) * sh2.row(0)(0) + sh1.row(2)(0) * sh2.row(0)(4)) - (sh1.row(0)(2) * sh2.row(4)(0) + sh1.row(0)(0) * sh2.row(4)(4)));
    sh3.row(2)(1) = sqrt( 8.0 /  5.0) * sh1.row(1)(1) * sh2.row(1)(0) + sqrt( 6.0 /  5.0) * sh1.row(0)(1) * sh2.row(2)(0) + -sqrt( 1.0 / 10.0) * (sh1.row(2)(1) * sh2.row(0)(0) - sh1.row(0)(1) * sh2.row(4)(0));
    sh3.row(2)(2) = sh1.row(1)(1) * sh2.row(1)(1) + sqrt( 3.0 /  4.0) * sh1.row(0)(1) * sh2.row(2)(1) + -sqrt( 1.0 / 16.0) * (sh1.row(2)(1) * sh2.row(0)(1) - sh1.row(0)(1) * sh2.row(4)(1));
    sh3.row(2)(3) = sqrt( 8.0 /  9.0) * sh1.row(1)(1) * sh2.row(1)(2) + sqrt( 2.0 /  3.0) * sh1.row(0)(1) * sh2.row(2)(2) + -sqrt( 1.0 / 18.0) * (sh1.row(2)(1) * sh2.row(0)(2) - sh1.row(0)(1) * sh2.row(4)(2));
    sh3.row(2)(4) = sh1.row(1)(1) * sh2.row(1)(3) + sqrt( 3.0 /  4.0) * sh1.row(0)(1) * sh2.row(2)(3) + -sqrt( 1.0 / 16.0) * (sh1.row(2)(1) * sh2.row(0)(3) - sh1.row(0)(1) * sh2.row(4)(3));
    sh3.row(2)(5) = sqrt( 8.0 /  5.0) * sh1.row(1)(1) * sh2.row(1)(4) + sqrt( 6.0 /  5.0) * sh1.row(0)(1) * sh2.row(2)(4) + -sqrt( 1.0 / 10.0) * (sh1.row(2)(1) * sh2.row(0)(4) - sh1.row(0)(1) * sh2.row(4)(4));
    sh3.row(2)(6) = sqrt( 4.0 / 15.0) * (sh1.row(1)(2) * sh2.row(1)(4) - sh1.row(1)(0) * sh2.row(1)(0)) + sqrt( 1.0 /  5.0) * (sh1.row(0)(2) * sh2.row(2)(4) - sh1.row(0)(0) * sh2.row(2)(0)) + -sqrt(1.0 / 60.0) * ((sh1.row(2)(2) * sh2.row(0)(4) - sh1.row(2)(0) * sh2.row(0)(0)) - (sh1.row(0)(2) * sh2.row(4)(4) - sh1.row(0)(0) * sh2.row(4)(0)));

    sh3.row(3)(0) = sqrt( 3.0 / 10.0) * (sh1.row(1)(2) * sh2.row(2)(0) + sh1.row(1)(0) * sh2.row(2)(4)) + -sqrt( 1.0 / 10.0) * ((sh1.row(2)(2) * sh2.row(3)(0) + sh1.row(2)(0) * sh2.row(3)(4)) + (sh1.row(0)(2) * sh2.row(1)(0) + sh1.row(0)(0) * sh2.row(1)(4)));
    sh3.row(3)(1) = sqrt( 9.0 /  5.0) * sh1.row(1)(1) * sh2.row(2)(0) + -sqrt( 3.0 /  5.0) * (sh1.row(2)(1) * sh2.row(3)(0) + sh1.row(0)(1) * sh2.row(1)(0));
    sh3.row(3)(2) = sqrt( 9.0 /  8.0) * sh1.row(1)(1) * sh2.row(2)(1) + -sqrt( 3.0 /  8.0) * (sh1.row(2)(1) * sh2.row(3)(1) + sh1.row(0)(1) * sh2.row(1)(1));
    sh3.row(3)(3) = sh1.row(1)(1) * sh2.row(2)(2) + -sqrt( 1.0 /  3.0) * (sh1.row(2)(1) * sh2.row(3)(2) + sh1.row(0)(1) * sh2.row(1)(2));
    sh3.row(3)(4) = sqrt( 9.0 /  8.0) * sh1.row(1)(1) * sh2.row(2)(3) + -sqrt( 3.0 /  8.0) * (sh1.row(2)(1) * sh2.row(3)(3) + sh1.row(0)(1) * sh2.row(1)(3));
    sh3.row(3)(5) = sqrt( 9.0 /  5.0) * sh1.row(1)(1) * sh2.row(2)(4) + -sqrt( 3.0 /  5.0) * (sh1.row(2)(1) * sh2.row(3)(4) + sh1.row(0)(1) * sh2.row(1)(4));
    sh3.row(3)(6) = sqrt( 3.0 / 10.0) * (sh1.row(1)(2) * sh2.row(2)(4) - sh1.row(1)(0) * sh2.row(2)(0)) + -sqrt( 1.0 / 10.0) * ((sh1.row(2)(2) * sh2.row(3)(4) - sh1.row(2)(0) * sh2.row(3)(0)) + (sh1.row(0)(2) * sh2.row(1)(4) - sh1.row(0)(0) * sh2.row(1)(0)));

    sh3.row(4)(0) = sqrt( 4.0 / 15.0) * (sh1.row(1)(2) * sh2.row(3)(0) + sh1.row(1)(0) * sh2.row(3)(4)) + sqrt( 1.0 /  5.0) * (sh1.row(2)(2) * sh2.row(2)(0) + sh1.row(2)(0) * sh2.row(2)(4)) + -sqrt(1.0 / 60.0) * ((sh1.row(2)(2) * sh2.row(4)(0) + sh1.row(2)(0) * sh2.row(4)(4)) + (sh1.row(0)(2) * sh2.row(0)(0) + sh1.row(0)(0) * sh2.row(0)(4)));
    sh3.row(4)(1) = sqrt( 8.0 /  5.0) * sh1.row(1)(1) * sh2.row(3)(0) + sqrt( 6.0 /  5.0) * sh1.row(2)(1) * sh2.row(2)(0) + -sqrt( 1.0 / 10.0) * (sh1.row(2)(1) * sh2.row(4)(0) + sh1.row(0)(1) * sh2.row(0)(0));
    sh3.row(4)(2) = sh1.row(1)(1) * sh2.row(3)(1) + sqrt( 3.0 /  4.0) * sh1.row(2)(1) * sh2.row(2)(1) + -sqrt( 1.0 / 16.0) * (sh1.row(2)(1) * sh2.row(4)(1) + sh1.row(0)(1) * sh2.row(0)(1));
    sh3.row(4)(3) = sqrt( 8.0 /  9.0) * sh1.row(1)(1) * sh2.row(3)(2) + sqrt( 2.0 /  3.0) * sh1.row(2)(1) * sh2.row(2)(2) + -sqrt( 1.0 / 18.0) * (sh1.row(2)(1) * sh2.row(4)(2) + sh1.row(0)(1) * sh2.row(0)(2));
    sh3.row(4)(4) = sh1.row(1)(1) * sh2.row(3)(3) + sqrt( 3.0 /  4.0) * sh1.row(2)(1) * sh2.row(2)(3) + -sqrt( 1.0 / 16.0) * (sh1.row(2)(1) * sh2.row(4)(3) + sh1.row(0)(1) * sh2.row(0)(3));
    sh3.row(4)(5) = sqrt( 8.0 /  5.0) * sh1.row(1)(1) * sh2.row(3)(4) + sqrt( 6.0 /  5.0) * sh1.row(2)(1) * sh2.row(2)(4) + -sqrt( 1.0 / 10.0) * (sh1.row(2)(1) * sh2.row(4)(4) + sh1.row(0)(1) * sh2.row(0)(4));
    sh3.row(4)(6) = sqrt( 4.0 / 15.0) * (sh1.row(1)(2) * sh2.row(3)(4) - sh1.row(1)(0) * sh2.row(3)(0)) + sqrt( 1.0 /  5.0) * (sh1.row(2)(2) * sh2.row(2)(4) - sh1.row(2)(0) * sh2.row(2)(0)) + -sqrt(1.0 / 60.0) * ((sh1.row(2)(2) * sh2.row(4)(4) - sh1.row(2)(0) * sh2.row(4)(0)) + (sh1.row(0)(2) * sh2.row(0)(4) - sh1.row(0)(0) * sh2.row(0)(0)));

    sh3.row(5)(0) = sqrt( 1.0 /  6.0) * (sh1.row(1)(2) * sh2.row(4)(0) + sh1.row(1)(0) * sh2.row(4)(4)) + sqrt( 1.0 /  6.0) * ((sh1.row(2)(2) * sh2.row(3)(0) + sh1.row(2)(0) * sh2.row(3)(4)) - (sh1.row(0)(2) * sh2.row(1)(0) + sh1.row(0)(0) * sh2.row(1)(4)));
    sh3.row(5)(1) = sh1.row(1)(1) * sh2.row(4)(0) + (sh1.row(2)(1) * sh2.row(3)(0) - sh1.row(0)(1) * sh2.row(1)(0));
    sh3.row(5)(2) = sqrt( 5.0 /  8.0) * sh1.row(1)(1) * sh2.row(4)(1) + sqrt( 5.0 /  8.0) * (sh1.row(2)(1) * sh2.row(3)(1) - sh1.row(0)(1) * sh2.row(1)(1));
    sh3.row(5)(3) = sqrt( 5.0 /  9.0) * sh1.row(1)(1) * sh2.row(4)(2) + sqrt( 5.0 /  9.0) * (sh1.row(2)(1) * sh2.row(3)(2) - sh1.row(0)(1) * sh2.row(1)(2));
    sh3.row(5)(4) = sqrt( 5.0 /  8.0) * sh1.row(1)(1) * sh2.row(4)(3) + sqrt( 5.0 /  8.0) * (sh1.row(2)(1) * sh2.row(3)(3) - sh1.row(0)(1) * sh2.row(1)(3));
    sh3.row(5)(5) = sh1.row(1)(1) * sh2.row(4)(4) + (sh1.row(2)(1) * sh2.row(3)(4) - sh1.row(0)(1) * sh2.row(1)(4));
    sh3.row(5)(6) = sqrt( 1.0 /  6.0) * (sh1.row(1)(2) * sh2.row(4)(4) - sh1.row(1)(0) * sh2.row(4)(0)) + sqrt( 1.0 /  6.0) * ((sh1.row(2)(2) * sh2.row(3)(4) - sh1.row(2)(0) * sh2.row(3)(0)) - (sh1.row(0)(2) * sh2.row(1)(4) - sh1.row(0)(0) * sh2.row(1)(0)));

    sh3.row(6)(0) = sqrt( 1.0 /  4.0) * ((sh1.row(2)(2) * sh2.row(4)(0) + sh1.row(2)(0) * sh2.row(4)(4)) - (sh1.row(0)(2) * sh2.row(0)(0) + sh1.row(0)(0) * sh2.row(0)(4)));
    sh3.row(6)(1) = sqrt( 3.0 /  2.0) * (sh1.row(2)(1) * sh2.row(4)(0) - sh1.row(0)(1) * sh2.row(0)(0));
    sh3.row(6)(2) = sqrt(15.0 / 16.0) * (sh1.row(2)(1) * sh2.row(4)(1) - sh1.row(0)(1) * sh2.row(0)(1));
    sh3.row(6)(3) = sqrt( 5.0 /  6.0) * (sh1.row(2)(1) * sh2.row(4)(2) - sh1.row(0)(1) * sh2.row(0)(2));
    sh3.row(6)(4) = sqrt(15.0 / 16.0) * (sh1.row(2)(1) * sh2.row(4)(3) - sh1.row(0)(1) * sh2.row(0)(3));
    sh3.row(6)(5) = sqrt( 3.0 /  2.0) * (sh1.row(2)(1) * sh2.row(4)(4) - sh1.row(0)(1) * sh2.row(0)(4));
    sh3.row(6)(6) = sqrt( 1.0 /  4.0) * ((sh1.row(2)(2) * sh2.row(4)(4) - sh1.row(2)(0) * sh2.row(4)(0)) - (sh1.row(0)(2) * sh2.row(0)(4) - sh1.row(0)(0) * sh2.row(0)(0)));
    return;
}

__device__ Eigen::Quaternionf Q_SlerpCUDA(Eigen::Quaternionf const &start_q, Eigen::Quaternionf &end_q, double t)
{
        Eigen::Quaternionf lerp_q;

        double cos_angle = start_q.x() * end_q.x()
                         + start_q.y() * end_q.y()
                         + start_q.z() * end_q.z()
                         + start_q.w() * end_q.w();

        if (cos_angle < 0) {
                end_q.x() = -end_q.x();
                end_q.y() = -end_q.y();
                end_q.z() = -end_q.z();
                end_q.w() = -end_q.w();
                cos_angle = -cos_angle;
        }

        double ratio_A, ratio_B;
        if (cos_angle > 0.99995f) {
                ratio_A = 1.0f - t;
                ratio_B = t;
        }
        else {
                double sin_angle = sqrt( 1.0f - cos_angle * cos_angle);
                double angle = atan2(sin_angle, cos_angle);
                ratio_A = sin((1.0f - t) * angle)  / sin_angle;
                ratio_B = sin(t * angle) / sin_angle;
        }

        lerp_q.x() = ratio_A * start_q.x() + ratio_B * end_q.x();
        lerp_q.y() = ratio_A * start_q.y() + ratio_B * end_q.y();
        lerp_q.z() = ratio_A * start_q.z() + ratio_B * end_q.z();
        lerp_q.w() = ratio_A * start_q.w() + ratio_B * end_q.w();

        return lerp_q.normalized();
}

__device__ void SH_RotationCUDA(Eigen::Matrix3f rots, float* Shs){
    Eigen::Matrix<float, 3, 3> sh1;
    Eigen::Matrix<float, 5, 5> sh2;
    Eigen::Matrix<float, 7, 7> sh3;
    Construct_SH_Rotation_MatrixCUDA(sh1, sh2, sh3, rots);

    Eigen::Matrix<float, 16, 1> r;
    Eigen::Matrix<float, 16, 1> g;
    Eigen::Matrix<float, 16, 1> b;

    for (int i = 0; i < 16; i++){
        if (i % 2 == 1){
            r.row(i)(0) = -Shs[i*3];
            g.row(i)(0) = -Shs[i*3 + 1];
            b.row(i)(0) = -Shs[i*3 + 2];
        }
        else{
            r.row(i)(0) = Shs[i*3];
            g.row(i)(0) = Shs[i*3 + 1];
            b.row(i)(0) = Shs[i*3 + 2];
        }
    }

    r.block<3, 1>(1, 0) = sh1*r.block<3, 1>(1, 0);
    g.block<3, 1>(1, 0) = sh1*g.block<3, 1>(1, 0);
    b.block<3, 1>(1, 0) = sh1*b.block<3, 1>(1, 0);

    r.block<5, 1>(4, 0) = sh2*r.block<5, 1>(4, 0);
    g.block<5, 1>(4, 0) = sh2*g.block<5, 1>(4, 0);
    b.block<5, 1>(4, 0) = sh2*b.block<5, 1>(4, 0);

    r.block<7, 1>(9, 0) = sh3*r.block<7, 1>(9, 0);
    g.block<7, 1>(9, 0) = sh3*g.block<7, 1>(9, 0);
    b.block<7, 1>(9, 0) = sh3*b.block<7, 1>(9, 0);

    for (int i = 0; i < 16; i++){
        if (i % 2 == 1){
            Shs[i*3] = -r.row(i)(0);
            Shs[i*3 + 1] = -g.row(i)(0);
            Shs[i*3 + 2] = -b.row(i)(0);
        }
        else{
            Shs[i*3] = r.row(i)(0);
            Shs[i*3 + 1] = g.row(i)(0);
            Shs[i*3 + 2] = b.row(i)(0);
        }
    }

    return; 
}

__global__ void RotateSHs(int sp_num, float* sample_neighbor_weights, int* sample_neighbor_nodes, Eigen::Quaternionf* q_vector_cuda, float* aim_feature_cuda, int k, int* static_samples_cuda){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= sp_num)
		return;
    if (static_samples_cuda[idx])
        return;
    
    Eigen::Quaternionf weighted_q(1.0, 0.0, 0.0, 0.0);
    float last_weight = 0.0;

    for (unsigned int j = 0; j < k; j++) {
        float cur_weight = sample_neighbor_weights[idx*k + j];
        float t = cur_weight / (cur_weight + last_weight);
        int node_idx = sample_neighbor_nodes[idx*k + j];
        weighted_q = Q_SlerpCUDA(weighted_q, q_vector_cuda[node_idx], t);

        last_weight += cur_weight;
    }

    Eigen::Matrix3f shs_rot = weighted_q.normalized().toRotationMatrix();
    SH_RotationCUDA(shs_rot, aim_feature_cuda + idx*48);
}

// 此处还有小于5ms的优化空间,用二分法找gs_idx
__global__ void FetchGsIdx(int* gs_containing_prefix_sum, int* paired_gs_idx, int total_containings, int gs_num){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= gs_num)
		return;
    int left = 0;
    int right = gs_containing_prefix_sum[0];
    if (idx > 0){
        left = gs_containing_prefix_sum[idx - 1];
        right = gs_containing_prefix_sum[idx];
    }
    for (int i = left; i < right; i++){
        paired_gs_idx[i] = idx;
    }
}

__global__ void FetchPairs(int* gs_containing_maximum_offset, int* gs_containing_prefix_sum, int* neighbouring_sp_idx, int* paired_gs_idx, int* paired_sp_idx, int total_containings){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= total_containings)
		return;
    int start = 0;
    if (paired_gs_idx[idx] > 0){
        start = gs_containing_prefix_sum[paired_gs_idx[idx] - 1];
    }
    int offset = idx - start;

    start = 0;
        if (paired_gs_idx[idx] > 0){
        start = gs_containing_maximum_offset[paired_gs_idx[idx] - 1];
    }
    paired_sp_idx[idx] = neighbouring_sp_idx[start + offset];
}

__global__ void EstimateMaxContaining(float* gs_3d_scale_max, int gs_num, int* gs_containing_maximum_offset, int estimated_coeff, int estimated_const){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= gs_num)
		return;
    if (gs_3d_scale_max[idx] == 0.0f){
        gs_containing_maximum_offset[idx] = 0;
    }
    else{
        gs_containing_maximum_offset[idx] = (int) ceil(pow(gs_3d_scale_max[idx]*estimated_coeff, 3.0f)) + estimated_const;
    }
}

__global__ void myKernelTest1(PointPlusPayload* data, int numData, float* gs_pos, float* gs_3d_scale_max, int gs_num, int* gs_containing_prefix_sum, int* gs_containing_maximum_offset, int* neighbouring_sp_idx){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= gs_num)
		return;
    if (gs_3d_scale_max[idx] == 0.0f){
        gs_containing_prefix_sum[idx] = 0;
        return;
    }
    int offset = 0;
    int max_count = gs_containing_maximum_offset[0];
    if (idx > 0){
        offset = gs_containing_maximum_offset[idx - 1];  
        max_count = gs_containing_maximum_offset[idx] - gs_containing_maximum_offset[idx - 1];
    }
    float3 queryPoint = {gs_pos[idx*3 + 0], gs_pos[idx*3 + 1], gs_pos[idx*3 + 2]};
    
    cukd::stackBased::radiusSearchCUDA<PointPlusPayload, cukd::PointPlusPayload_traits>(queryPoint, data, numData, pow(gs_3d_scale_max[idx], 2.0f), neighbouring_sp_idx + offset, *(gs_containing_prefix_sum + idx), max_count);
}

__global__ void HighlightSelectedGs(int gs_num, float* shs_highlight_cuda, int* gs_type_cuda, bool during_deforming, int shs_dim, int knn_k) {
    auto idx = cg::this_grid().thread_rank();
	if (idx >= gs_num)
		return;

    float color_ratio = ((float) gs_type_cuda[idx]) / (float)(knn_k*6.0f);
    
    if (color_ratio > 1.0f){
        float color_ratio_ = color_ratio / (float)(knn_k * 6 * 2);
        shs_highlight_cuda[idx*shs_dim + 0] += 2.0*color_ratio_;
        shs_highlight_cuda[idx*shs_dim + 1] += 2.0*color_ratio_;
        shs_highlight_cuda[idx*shs_dim + 2] -= 2.0*color_ratio_;
    }
    else if (color_ratio > 1e-5f){
        if (during_deforming){
            shs_highlight_cuda[idx*shs_dim + 0] -= 1.0*color_ratio;
            shs_highlight_cuda[idx*shs_dim + 1] += 1.0*color_ratio;
            shs_highlight_cuda[idx*shs_dim + 2] -= 1.0*color_ratio;
        }
        else {
            shs_highlight_cuda[idx*shs_dim + 0] -= 2.0*color_ratio;
            shs_highlight_cuda[idx*shs_dim + 1] += 2.0*color_ratio;
            shs_highlight_cuda[idx*shs_dim + 2] -= 2.0*color_ratio;
        }
    }
    else if (color_ratio < -1e-5f){
        shs_highlight_cuda[idx*shs_dim + 0] -= 2.0*color_ratio;
        shs_highlight_cuda[idx*shs_dim + 1] += 2.0*color_ratio;
        shs_highlight_cuda[idx*shs_dim + 2] += 2.0*color_ratio;
    }
} 

void BuildTreeCUDA(PointPlusPayload* data, int numData){
    
    cukd::box_t<float3> *d_bounds;
    cudaMallocManaged((void**)&d_bounds,sizeof(cukd::box_t<float3>));
    cudaDeviceSynchronize();


    // auto start = std::chrono::steady_clock::now();
    cukd::buildTree<PointPlusPayload, cukd::PointPlusPayload_traits>(data, numData, d_bounds);
    // std::cout << "GPU samples count = " << numData << std::endl;
    cudaDeviceSynchronize();
    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsedSeconds = end - start;
    // std::cout << "gpu kdtree took " << elapsedSeconds.count() << " seconds." << std::endl;
    
    cudaFree(d_bounds);
    return;
}

void EstimateCUDA(float* gs_3d_scale_max, int gs_num, int* gs_containing_maximum_offset, int& total_estimated_containings, int estimated_coeff, int estimated_const){
    size_t scan_size0;
    char* scan_temp_storage0;

    // auto start = std::chrono::steady_clock::now();
    EstimateMaxContaining<<<(gs_num + 1023) / 1024, 1024 >>>(gs_3d_scale_max, gs_num, gs_containing_maximum_offset, estimated_coeff, estimated_const);
    cudaDeviceSynchronize();
    cub::DeviceScan::InclusiveSum(nullptr, scan_size0, gs_containing_maximum_offset, gs_containing_maximum_offset, gs_num);
    cudaDeviceSynchronize();
    cudaMallocManaged((void**)&scan_temp_storage0, scan_size0);
    cudaDeviceSynchronize();
    cub::DeviceScan::InclusiveSum(scan_temp_storage0, scan_size0, gs_containing_maximum_offset, gs_containing_maximum_offset, gs_num);
    cudaDeviceSynchronize();

    cudaMemcpy(&total_estimated_containings, gs_containing_maximum_offset + (gs_num-1), sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Estimated total containings " << total_estimated_containings << std::endl;

    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsedSeconds = end - start;
    // std::cout << "EstimateCUDA took " << elapsedSeconds.count() << " seconds." << std::endl;

    cudaFree(scan_temp_storage0);
}


void QueryTreeCUDA(PointPlusPayload* data, int numData, float* gs_pos, float* gs_3d_scale_max, int gs_num, int* gs_containing_prefix_sum, int* gs_containing_maximum_offset, int* neighbouring_sp_idx, int& total_containings){
    // auto start = std::chrono::steady_clock::now();
    myKernelTest1<<<(gs_num + 1023) / 1024, 1024 >>>(data , numData, gs_pos, gs_3d_scale_max, gs_num, gs_containing_prefix_sum, gs_containing_maximum_offset, neighbouring_sp_idx);
    cudaDeviceSynchronize();
    
    size_t scan_size1;
    char* scan_temp_storage1;

    cub::DeviceScan::InclusiveSum(nullptr, scan_size1, gs_containing_prefix_sum, gs_containing_prefix_sum, gs_num);
    cudaDeviceSynchronize();
    cudaMallocManaged((void**)&scan_temp_storage1, scan_size1);
    cudaDeviceSynchronize();
    cub::DeviceScan::InclusiveSum(scan_temp_storage1, scan_size1, gs_containing_prefix_sum, gs_containing_prefix_sum, gs_num);
    cudaDeviceSynchronize();

    cudaMemcpy(&total_containings, gs_containing_prefix_sum + (gs_num-1), sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // std::cout << "Total containings " << total_containings << std::endl;

    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsedSeconds = end - start;
    // std::cout << "QueryTreeCUDA took " << elapsedSeconds.count() << " seconds." << std::endl;

    cudaFree(scan_temp_storage1);
}


void GetPairCUDA(int gs_num, int* gs_containing_prefix_sum, int* gs_containing_maximum_offset, int* paired_gs_idx, int* paired_sp_idx, int& total_containings, int* neighbouring_sp_idx){
    // auto start = std::chrono::steady_clock::now();

    FetchGsIdx<<<(gs_num + 1023) / 1024, 1024 >>>(gs_containing_prefix_sum, paired_gs_idx, total_containings, gs_num);
    cudaDeviceSynchronize();
    FetchPairs<<<(total_containings + 1023) / 1024, 1024 >>>(gs_containing_maximum_offset, gs_containing_prefix_sum, neighbouring_sp_idx, paired_gs_idx, paired_sp_idx, total_containings);
    cudaDeviceSynchronize();
    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsedSeconds = end - start;
    // std::cout << "GetPairCUDA took " << elapsedSeconds.count() << " seconds." << std::endl;
}

__global__ void GetGsGrid(int gs_num, float grid_step, int grid_num, float* gs_pos, int* grid_gs_count, float3 min_xyz){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= gs_num)
		return;

    int x_idx = int(floor((gs_pos[idx*3 + 0] - min_xyz.x)/grid_step));
    int y_idx = int(floor((gs_pos[idx*3 + 1] - min_xyz.y)/grid_step));
    int z_idx = int(floor((gs_pos[idx*3 + 2] - min_xyz.z)/grid_step));

    int grid_idx = x_idx*grid_num*grid_num + y_idx*grid_num + z_idx;
    if (grid_idx < 0){
        printf("overflow occurs x_idx: %d, y_idx: %d, z_idx: %d", x_idx, y_idx, z_idx);
        printf("position: %d, %d, %d", gs_pos[idx*3 + 0], gs_pos[idx*3 + 1], gs_pos[idx*3 + 2]);
    }
    atomicAdd(&grid_gs_count[grid_idx], 1);
    // if (grid_gs_count[grid_idx] > 5){
    //     printf("grid_idx %d, grid_gs_count %d", grid_idx, grid_gs_count[grid_idx]);
    // }

    return;
}

__global__ void GetBoxesGsGrid(int gs_num, float grid_step, int grid_num, float* gs_aabbs_cuda, int* grid_gs_count, float3 min_xyz, int padding){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= gs_num)
		return;

    // float max_side = max(gs_aabbs_cuda[idx*6 + 3] - gs_aabbs_cuda[idx*6 + 0], gs_aabbs_cuda[idx*6 + 4] - gs_aabbs_cuda[idx*6 + 1]);
    // max_side = max(max_side, gs_aabbs_cuda[idx*6 + 5] - gs_aabbs_cuda[idx*6 + 2]);

    int x_idx_min = max(int(floor((gs_aabbs_cuda[idx*6 + 0] - min_xyz.x)/grid_step) - padding), 0);
    int y_idx_min = max(int(floor((gs_aabbs_cuda[idx*6 + 1] - min_xyz.y)/grid_step) - padding), 0);
    int z_idx_min = max(int(floor((gs_aabbs_cuda[idx*6 + 2] - min_xyz.z)/grid_step) - padding), 0);
    
    int x_idx_max = min(int(floor((gs_aabbs_cuda[idx*6 + 3] - min_xyz.x)/grid_step) + padding), grid_num-1);
    int y_idx_max = min(int(floor((gs_aabbs_cuda[idx*6 + 4] - min_xyz.y)/grid_step) + padding), grid_num-1);
    int z_idx_max = min(int(floor((gs_aabbs_cuda[idx*6 + 5] - min_xyz.z)/grid_step) + padding), grid_num-1);

    for (int x_idx = x_idx_min; x_idx < x_idx_max + 1; x_idx++){
        for (int y_idx = y_idx_min; y_idx < y_idx_max + 1; y_idx++){
            for (int z_idx = z_idx_min; z_idx < z_idx_max + 1; z_idx++){
                int grid_idx = x_idx*grid_num*grid_num + y_idx*grid_num + z_idx;
                atomicAdd(&grid_gs_count[grid_idx], 1);
            }
        }
    }

    return;
}

__global__ void GetGridsNumPerGs(int gs_num, float grid_step, int grid_num, float* gs_aabbs_cuda, int* per_gs_grids_cuda, float3 min_xyz, int padding){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= gs_num)
		return;

    int x_idx_min = max(int(floor((gs_aabbs_cuda[idx*6 + 0] - min_xyz.x)/grid_step) - padding), 0);
    int y_idx_min = max(int(floor((gs_aabbs_cuda[idx*6 + 1] - min_xyz.y)/grid_step) - padding), 0);
    int z_idx_min = max(int(floor((gs_aabbs_cuda[idx*6 + 2] - min_xyz.z)/grid_step) - padding), 0);
    
    int x_idx_max = min(int(floor((gs_aabbs_cuda[idx*6 + 3] - min_xyz.x)/grid_step) + padding), grid_num-1);
    int y_idx_max = min(int(floor((gs_aabbs_cuda[idx*6 + 4] - min_xyz.y)/grid_step) + padding), grid_num-1);
    int z_idx_max = min(int(floor((gs_aabbs_cuda[idx*6 + 5] - min_xyz.z)/grid_step) + padding), grid_num-1);

    per_gs_grids_cuda[idx] = (x_idx_max + 1 - x_idx_min)*(y_idx_max + 1 - y_idx_min)*(z_idx_max + 1 - z_idx_min);

    return;
}


__global__ void GetGsGridPair(int gs_num, float grid_step, int grid_num, float* gs_aabbs_cuda, int* per_gs_grids_cuda, float3 min_xyz, int padding, int* grid_index_cuda, int* gaussian_index_cuda){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= gs_num)
		return;


    int x_idx_min = max(int(floor((gs_aabbs_cuda[idx*6 + 0] - min_xyz.x)/grid_step) - padding), 0);
    int y_idx_min = max(int(floor((gs_aabbs_cuda[idx*6 + 1] - min_xyz.y)/grid_step) - padding), 0);
    int z_idx_min = max(int(floor((gs_aabbs_cuda[idx*6 + 2] - min_xyz.z)/grid_step) - padding), 0);
    
    int x_idx_max = min(int(floor((gs_aabbs_cuda[idx*6 + 3] - min_xyz.x)/grid_step) + padding), grid_num-1);
    int y_idx_max = min(int(floor((gs_aabbs_cuda[idx*6 + 4] - min_xyz.y)/grid_step) + padding), grid_num-1);
    int z_idx_max = min(int(floor((gs_aabbs_cuda[idx*6 + 5] - min_xyz.z)/grid_step) + padding), grid_num-1);

    int start_idx;
    if (idx == 0){
        start_idx = 0;
    }
    else {
        start_idx = per_gs_grids_cuda[idx-1];
    }

    int offset = 0;
    for (int x_idx = x_idx_min; x_idx < x_idx_max + 1; x_idx++){
        for (int y_idx = y_idx_min; y_idx < y_idx_max + 1; y_idx++){
            for (int z_idx = z_idx_min; z_idx < z_idx_max + 1; z_idx++){
                int grid_idx = x_idx*grid_num*grid_num + y_idx*grid_num + z_idx;
                gaussian_index_cuda[start_idx + offset] = idx;
                grid_index_cuda[start_idx + offset] = grid_idx;
                offset += 1;
            }
        }
    }
    return;
}

void GetGsNumPerGridCUDA(int gs_num, float grid_step, int grid_num, float* gs_pos, int* grid_gs_count, float3 min_xyz){
    GetGsGrid<<<(gs_num + 1023) / 1024, 1024 >>>(gs_num, grid_step, grid_num, gs_pos, grid_gs_count, min_xyz);
    cudaDeviceSynchronize();

    int grid_count = grid_num*grid_num*grid_num;

    size_t scan_size1;
    char* scan_temp_storage1;

    cub::DeviceScan::InclusiveSum(nullptr, scan_size1, grid_gs_count, grid_gs_count, grid_count);
    cudaDeviceSynchronize();
    cudaMallocManaged((void**)&scan_temp_storage1, scan_size1);
    cudaDeviceSynchronize();
    cub::DeviceScan::InclusiveSum(scan_temp_storage1, scan_size1, grid_gs_count, grid_gs_count, grid_count);
    cudaDeviceSynchronize();

    cudaFree(scan_temp_storage1);

    // std::cout << "GET GAUSSIAN PER GRID" << std::endl;
    return;
}




void GetBoxedGsNumPerGridCUDA(int gs_num, float grid_step, int grid_num, float* gs_aabbs_cuda, int* grid_gs_count, float3 min_xyz, int padding){
    GetBoxesGsGrid<<<(gs_num + 1023) / 1024, 1024 >>>(gs_num, grid_step, grid_num, gs_aabbs_cuda, grid_gs_count, min_xyz, padding);
    cudaDeviceSynchronize();

    int grid_count = grid_num*grid_num*grid_num;

    size_t scan_size1;
    char* scan_temp_storage1;

    cub::DeviceScan::InclusiveSum(nullptr, scan_size1, grid_gs_count, grid_gs_count, grid_count);
    cudaDeviceSynchronize();
    cudaMallocManaged((void**)&scan_temp_storage1, scan_size1);
    cudaDeviceSynchronize();
    cub::DeviceScan::InclusiveSum(scan_temp_storage1, scan_size1, grid_gs_count, grid_gs_count, grid_count);
    cudaDeviceSynchronize();

    cudaFree(scan_temp_storage1);

    std::cout << "GET GAUSSIAN PER GRID" << std::endl;
    return;
}

void GetGsGridPairCUDA(int gs_num, float grid_step, int grid_num, float* gs_aabbs_cuda, int* per_gs_grids_cuda, float3 min_xyz, int padding, int* grid_index_cuda, int* gaussian_index_cuda){
    GetGsGridPair<<<(gs_num + 1023) / 1024, 1024 >>>(gs_num, grid_step, grid_num, gs_aabbs_cuda, per_gs_grids_cuda, min_xyz, padding, grid_index_cuda, gaussian_index_cuda);
    cudaDeviceSynchronize();

    return;
}

void GetGridsNumPerGsCUDA(int gs_num, float grid_step, int grid_num, float* gs_aabbs_cuda, int* per_gs_grids_cuda, float3 min_xyz, int padding){
    GetGridsNumPerGs<<<(gs_num + 1023) / 1024, 1024 >>>(gs_num, grid_step, grid_num, gs_aabbs_cuda, per_gs_grids_cuda, min_xyz, padding);
    cudaDeviceSynchronize();

    size_t scan_size1;
    char* scan_temp_storage1;

    cub::DeviceScan::InclusiveSum(nullptr, scan_size1, per_gs_grids_cuda, per_gs_grids_cuda, gs_num);
    cudaDeviceSynchronize();
    cudaMallocManaged((void**)&scan_temp_storage1, scan_size1);
    cudaDeviceSynchronize();
    cub::DeviceScan::InclusiveSum(scan_temp_storage1, scan_size1, per_gs_grids_cuda, per_gs_grids_cuda, gs_num);
    cudaDeviceSynchronize();

    cudaFree(scan_temp_storage1);

    std::cout << "GET Grids Num PER Gaussian" << std::endl;
    return;
}


void RotateSamplesSHsCUDA(int sp_num, float* sample_neighbor_weights, int* sample_neighbor_nodes, Eigen::Quaternionf* q_vector_cuda, float* aim_feature_cuda, int k, int* static_samples_cuda){

    RotateSHs<<<(sp_num + 255) / 256, 256>>>(sp_num, sample_neighbor_weights, sample_neighbor_nodes, q_vector_cuda, aim_feature_cuda, k, static_samples_cuda);
    cudaDeviceSynchronize();

    return;

}

void PredictSamplesPosCUDA(int sp_num, float* sample_neighbor_weights, int* sample_neighbor_nodes, float* rots_cuda, float* trans_cuda, float* sample_positions, int k){

    return;
} 

void HighlightSelectedGsCUDA(int gs_num, float* shs_highlight_cuda, int* gs_type_cuda, bool during_deforming, int shs_dim, int knn_k){
    HighlightSelectedGs<<<(gs_num + 1023) / 1024, 1024>>>(gs_num, shs_highlight_cuda, gs_type_cuda, during_deforming, shs_dim, knn_k);
    return;
}


		// if (! CheckRotationContinue(q, origin_q)){
		// 	std::cout << "P: " << P << std::endl;
		// 	std::cout << "M*Q: " << M*Q << "\n" << std::endl;

		// 	Eigen::Matrix3f K_mat;
		// 	K_mat.setZero();
		// 	K_mat.diagonal() = K;

		// 	Eigen::MatrixXf M_rc(4, 4);
		// 	M_rc = M;
		// 	M_rc.block<3, 3>(0, 0) = (dest_rot_o*K_mat).transpose();
		// 	std::cout << "M_rc*Q: " << M_rc*Q << "\n" << std::endl;
			

		// 	std::cout << "A: " << dest_rot << std::endl;
		// 	std::cout << "R*S: " << dest_rot_o*S << std::endl;
		// 	std::cout << "R*K: " << dest_rot_o*K_mat << "\n" << std::endl;
		// 	for (int i = 0; i < 3; i++) {
		// 		scale_vector[gs_idx].scale[i] = 1e-3;
		// 	}
		// 	opacity_vector[gs_idx] = 1e-3;
		// }
        