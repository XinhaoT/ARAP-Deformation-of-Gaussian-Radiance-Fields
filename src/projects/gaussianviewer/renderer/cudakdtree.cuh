#ifndef CUDA_KDTREE_CUH
#define CUDA_KDTREE_CUH
#pragma once
#include "helper.hpp"
#include "cukd/point_struct.h"
// #include "cukd/radius.h"


void BuildTreeCUDA(PointPlusPayload* data, int numData);

void EstimateCUDA(float* gs_3d_scale_max, int gs_num, int* gs_containing_maximum_offset, int& total_estimated_containings, int estimated_coeff, int estimated_const);

void QueryTreeCUDA(PointPlusPayload* data, int numData, float* gs_pos, float* gs_3d_scale_max, int gs_num, int* gs_containing_prefix_sum, int* gs_containing_maximum_offset, int* neighbouring_sp_idx, int& total_containings);

void GetPairCUDA(int gs_num, int* gs_containing_prefix_sum, int* gs_containing_maximum_offset, int* paired_gs_idx, int* paired_sp_idx, int& total_containings, int* neighbouring_sp_idx);

void GetGsNumPerGridCUDA(int gs_num, float grid_step, int grid_num, float* gs_pos, int* grid_gs_count, float3 min_xyz);

void GetBoxedGsNumPerGridCUDA(int gs_num, float grid_step, int grid_num, float* gs_aabbs_cuda, int* grid_gs_count, float3 min_xyz, int padding);

void GetGridsNumPerGsCUDA(int gs_num, float grid_step, int grid_num, float* gs_aabbs_cuda, int* per_gs_grids_cuda, float3 min_xyz, int padding);

void GetGsGridPairCUDA(int gs_num, float grid_step, int grid_num, float* gs_aabbs_cuda, int* per_gs_grids_cuda, float3 min_xyz, int padding, int* grid_index_cuda, int* gaussian_index_cuda);

void RotateSamplesSHsCUDA(int sp_num, float* sample_neighbor_weights, int* sample_neighbor_nodes, Eigen::Quaternionf* q_vector_cuda, float* aim_feature_shs, int k, int* static_samples_cuda);

void PredictSamplesPosCUDA(int sp_num, float* sample_neighbor_weights, int* sample_neighbor_nodes, float* rots_cuda, float* trans_cuda, float* sample_positions, int k);

void HighlightSelectedGsCUDA(int gs_num, float* shs_highlight_cuda, int* gs_type_cuda, bool during_deforming, int shs_dim, int knn_k);

#endif
