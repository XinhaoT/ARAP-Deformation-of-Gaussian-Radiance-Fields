#ifndef SNAPSHOT_H
#define SNAPSHOT_H

#include "helper.hpp"
#include "Samples.hpp"
#include "Deform.hpp"
#include "cudakdtree.cuh"
class Snapshot{
    public:

        int gs_num;
        std::vector<Pos> pos_vector;
        std::vector<Rot> rot_vector;
        std::vector<Vertex> vertices;
        std::vector<float> opacity_vector;
        std::vector<SHs<3>> shs_vector;
        std::vector<Scale> scale_vector;

        std::vector<Scale> scale_3d_clip;
		std::vector<float> scale_3d_max;

        int sp_num;
        vector<Pos> sample_positions;
        vector<PointPlusPayload> samples_pos_payload_vector;
        SamplePoints sps;


        DeformGraph deform_graph;
		vector<Pos>    aim_centers;
        vector<float>  aim_centers_radius;
        set<unsigned int> static_indices;
    	set<unsigned int> control_indices;
		vector<vector<unsigned int>> indices_blocks;
        vector<int> blocks_type;
		int block_num;

    private:

};

using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;
using namespace std;

#endif