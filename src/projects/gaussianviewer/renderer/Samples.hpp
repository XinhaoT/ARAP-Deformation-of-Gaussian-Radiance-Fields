#ifndef SAMPLES_H
#define SAMPLES_H

#include "helper.hpp"

using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;
using namespace std;



class SamplePoints{
public:
    vector<SamplePoint> sample_points;
    vector<SamplePoint> backup_sample_points;

    vector<vector<int>> neighbourgs;
    vector<int> neighbourgs_offsets;
    vector<vector<float>> neighbour_pdfs;

    int total_neighbouring;

    Vector1d m_x;
    std::vector<Vertex> vertices;
    vector<Pos> positions;
    std::vector<float> opacity_gs;
    std::vector<SHs<3>> shs_gs;
    std::vector<SHs<3>> shs_aim;

    SamplePoints(){}

    SamplePoints(vector<SamplePoint> sample_points){
        this->sample_points = sample_points;
        // for (int i = 0; i < sample_points.size(); i++) {
        //     backup_sample_points.push_back(sample_points[i]);
        // }
    }

    void reset(){
        // sample_points.clear();
        // for (int i = 0; i < backup_sample_points.size(); i++) {
        //     sample_points.push_back(backup_sample_points[i]);
        // }
    }

    void construct(std::vector<Vertex> verts, vector<Pos> pos, std::vector<float> opacity_vector, std::vector<SHs<3>> shs_vector, vector<SHs<3>> sample_feature_shs);

    void gs_adjust();
    void initializeX();
    double optimize();
    void DefineJacobiStructure(SpMat& jacobi, SpMat& jacobiT);
    void FastCalcJacobiMat(SpMat& jacobi, SpMat& jacobiT);
    void CalcEnergyFunc(Vector1d& fx);

private:
    SpMat m_jacobi;
	SpMat m_jacobiT;
    int jacobi_m,jacobi_n,fx_n;
};

#endif