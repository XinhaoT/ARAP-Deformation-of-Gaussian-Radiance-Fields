#ifndef HELPERS_H
#define HELPERS_H
#include <string>
#include <vector>
#include <array>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>

// #include <torch/torch.h>

#include "Config.hpp"
#include <core/raycaster/KdTree.hpp>
#include <core/renderer/RenderMaskHolder.hpp>
#include <core/scene/BasicIBRScene.hpp>
#include <core/system/SimpleTimer.hpp>
#include <core/system/Config.hpp>
// #include <core/system/Quaternion.hpp>
#include <core/graphics/Mesh.hpp>
#include <core/graphics/MaterialMesh.hpp>
#include <core/view/ViewBase.hpp>
#include <core/renderer/CopyRenderer.hpp>
#include <core/renderer/PointBasedRenderer.hpp>
#include <memory>
#include <core/graphics/Texture.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <functional>
#include <rasterizer.h>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <glm/glm.hpp>
#include <chrono>
#include <omp.h>
#include "tiny_obj_loader.h"

#include "GaussianSurfaceRenderer.hpp"


#define Pi 3.1415926535
#define KNN_MAX 12

#define PDF_THRESHOLD 255.0
#define GS_HALF_LEN_SCALE 1.5
#define NUM_SAMPLES_PER_DIM 128
#define LOCAL_SAMPLES_PER_DIM 48
#define LOCAL_UNIFORM_DIST 0.002

#define SAMPLES_PER_GRID 4

#define OPACITY_LR 0.05
#define SHS_LR 0.0025
#define SH0 0.28209479

#define CUTOFF_ALPHA 1.0f/255.0f

#define DO_PDF_CLIP false

using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;
using namespace std;

typedef Vector3d Vec3;
typedef Vector4d Vec4;
typedef Matrix3d Mat3;
typedef Matrix4d Mat4;

typedef Eigen::Matrix<double,-1,1> Vector1d;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMat;

typedef sibr::Vector3f Pos;

template<int D>
struct SHs
{
	float shs[(D+1)*(D+1)*3];
};

struct Scale
{
	float scale[3];
};

struct Rot
{
	float rot[4];
};

template<int D>
struct RichPoint_Soup
{
	Pos pos;
	float n[3];
	SHs<D> shs;
	float opacity;
	Scale scale;
	Rot rot;
    //Edit
    float index;
};

template<int D>
struct RichPoint
{
	Pos pos;
	float n[3];
	SHs<D> shs;
	float opacity;
	Scale scale;
	Rot rot;
};


struct Vertex {
    unsigned int index;

    unsigned int Neighbor_Nodes[KNN_MAX];

    std::array<double, KNN_MAX> Neighbor_Weights;

	float half_length;

	float pdf_coeff;

	Eigen::Matrix3f sigma;
};

struct Node {
    Pos Position;

    Pos Color;

    unsigned int Neighbor[KNN_MAX];

    unsigned int Vertex_index;
};

struct SamplePoint {
    unsigned int aim_index;

    unsigned int Neighbor_Nodes[KNN_MAX];

    std::array<double, KNN_MAX> Neighbor_Weights;
};


struct AABB {
    Pos xyz_min;
    Pos xyz_max;
};
struct Vertex_m {
    float x, y, z;
};

struct Normal_m {
    float nx, ny, nz;
};

struct Face_m {
    int v1, v2, v3;
    int vn1, vn2, vn3;
};

class DeformHistory {
    public:
    DeformHistory(){}

    int total_operations = 0;
    int move_operations = 0;

    vector<int> operation_types; // 0 for Adding a block
                                 // 1 for common deformations
                                 // 2 for twisting operations
                                 // 3 for scaling operations
                                 // 4 for deformations with energy on each selected node
                                 // negative numbers for deletion blocks (deleting i-th block: -(i+1))

    vector<vector<unsigned int>> block_nodes;


    vector<vector<Pos>> mouse_movements;
    vector<vector<int>> blocks_types_moves;
                                    //record the blocks_type at each movements

    vector<int> energy_on_centers;  // 0 for seperate energy
                                    // 1 for centered energy

    vector<Vector4f> twist_axis;

};

class DeformScript {    // for some simple but precise operations
    public:
    DeformScript(){}

    vector<vector<unsigned int>> block_nodes;
    vector<int> blocks_types;

    vector<vector<Pos>> node_aims;
};

struct EndPoint {
    std::array<int, 3> axises; // the axis from the largest to smallest scale
    std::array<std::pair<Pos, Pos>, 3> ends;
};


void readObjFile(const std::string& filename, std::vector<Vertex_m>& vertices, std::vector<Normal_m>& normals, std::vector<Face_m>& faces);


sibr::Vector2f get_2d_pos(Pos pos_3d, sibr::Matrix4f T, const unsigned int width, const unsigned int height);
float pts_distance(const Pos pos1, const Pos pos2);
void random_sample_once(const std::vector<Pos> &vertices, std::vector<int> &node_idxs, float threshold);
std::vector<int> sample_control_points(const std::vector<Pos> &vertices, int node_num, float threshold_d, bool only_surface, const std::vector<int> &is_surface_vector);
std::vector<int> farthest_control_points_sampling(const std::vector<Pos> pos_vector, int node_num, bool only_surface);
bool hack_node_judgement(Pos pos);

void storeVectorToFile(const std::vector<int>& data, const std::string& filename);
void saveVectorToFile(const std::vector<float>& values, const std::string& filename);
void saveVectorToFile(const std::vector<SHs<3>>& values, const std::string& filename);
void saveVectorToFile(const std::vector<Scale>& values, const std::string& filename);
void saveVectorToFile(const std::vector<Rot>& values, const std::string& filename);
void saveVectorToFile(const std::vector<Pos>& values, const std::string& filename);
void saveEigenMatrixVectorToFile(const std::vector<Eigen::Matrix3f>& matrixVector, const std::string& filename);
void saveEigenVectorToTxt(const std::vector<Pos>& vec, const std::string& filename);
void LoadMeshPoints(const std::string& filename, std::vector<Pos> &meshPoints);
std::vector<int> loadVectorFromFile(const std::string& filename);

std::vector<float> readFloatsFromFile(const std::string& filename);
Eigen::Vector3f getSingularValues(Eigen::Matrix3f mat);
Eigen::Matrix3f getOthogonalMatrix(Eigen::Matrix3f mat);
Eigen::Matrix3f getOthogonalMatrixWithCheck(Eigen::Matrix3f mat);
Eigen::Matrix3f getOthogonalMatrixWithK(Eigen::Matrix3f mat,Eigen::Vector3f& K);
Eigen::Matrix3f FastgetOthogonalMatrix(Eigen::Matrix3f mat);
Eigen::Matrix3f FastgetOthogonalMatrixWithK(Eigen::Matrix3f mat, Eigen::Vector3f& K);
Eigen::Matrix3f FastgetOthogonalMatrixAndK(Eigen::Matrix3f mat, Eigen::Vector3f& K, Eigen::Matrix3f R0);
Eigen::Matrix3f FastgetK(Eigen::Matrix3f Affine, Eigen::Matrix3f R, Eigen::Vector3f& K, Eigen::Matrix3f R0);
bool CheckReconstructionError(Eigen::Matrix3f mat, Eigen::Matrix3f R, Eigen::Vector3f K);
bool CheckRotationContinue(Eigen::Quaternionf q0, Eigen::Quaternionf last_q);
bool CheckRotation(Eigen::Matrix3f R0, Eigen::Matrix3f R1);
Eigen::Matrix3f CastRot2Matrix(std::array<double, 9> rot);
Eigen::Quaternionf Quaternion_S_lerp(Eigen::Quaternionf const &start_q, Eigen::Quaternionf &end_q, double t);
Eigen::Vector3f spectDecomp(Eigen::Matrix3f S, Eigen::Matrix3f& U);
Eigen::Quaternionf snuggle(Eigen::Quaternionf q, Eigen::Vector3f& k);
void RunTestVKVt(Scale& s);
void RunTestQ();
Eigen::Vector3f GetKfromSR(Eigen::Matrix3f S, Eigen::Matrix3f R0);
void Construct_SH_Rotation_Matrix(Eigen::Matrix<float, 3, 3>& sh1, Eigen::Matrix<float, 5, 5>& sh2, Eigen::Matrix<float, 7, 7>& sh3, Eigen::Matrix<float, 3, 3> rots);
void SH_Rotation(Eigen::Matrix3f rots, std::vector<float>& Shs);
// void chooseDevice();
Pos PointRotateByAxis(Pos point, Pos center, Vector4f axis, float radian);
void Test_SH_Rotation();
std::array<int, 3> OrderIndices(Scale s);
void polarDecomposition(const Eigen::Matrix3f A, Eigen::Matrix3f& S);
int FetchFirstNodeIdx(vector<Pos> node_postions);

void writeVectorToObj(const std::vector<Pos>& vertices, const std::string& filename);
#endif



