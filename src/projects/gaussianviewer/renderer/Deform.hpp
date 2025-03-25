#ifndef DEFORM_H
#define DEFORM_H

#include "helper.hpp"

using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;
using namespace std;



class DeformGraph {
public:
    // node Data
    vector<Node>       nodes;
    vector<Node>       back_up_nodes;
    vector<unsigned int> indices;

    set<unsigned int> free_indices;
    set<unsigned int> static_indices;
    set<unsigned int> control_indices;

    vector<vector<unsigned int>> indices_blocks;
    vector<Pos> block_centers;

    unsigned int current_control_idx;

    /*
        rot mat:
        [   0   3   6   ]
        [   1   4   7   ]
        [   2   5   8   ]
        trans vec:
        [   tx  ty  tz  ]
    */
    vector<array<double, 9>> rot;           
    vector<array<double, 3>> trans;


    vector<vector<array<double, 9>>> history_rot;
    vector<vector<array<double, 3>>> history_trans;
    vector<vector<Pos>> history_node_positions;

    double w_rot;
    double w_reg;
    double w_con;

    int k_nearest;

    DeformGraph(){}

    // constructor
    DeformGraph(vector<Node> nodes, int knn_k):
	 k_nearest(knn_k),
	 w_rot(1.0),
	 w_reg(10.0),
	 w_con(100.0)
    {
        this->nodes = nodes;

        for (unsigned int i = 0; i < nodes.size(); i++){
            this->nodes[i].Color = Pos(0.6f, 0.0f, 0.0f);
            free_indices.insert(i);
        }

        vector<Node> back_up(this->nodes);
        this->back_up_nodes = back_up;

        for (int j = 0; j < nodes.size(); j++){
            array<double, 9> rotation;
            array<double, 3> translation;
            for(int i=0; i<9; i++) rotation[i] = 0.0;
            rotation[0] = 1.0; rotation[4] = 1.0; rotation[8] = 1.0;
            for(int i=0; i<3; i++) translation[i] = 0.0;
            rot.push_back(rotation);
            trans.push_back(translation);
	    }

        setupEdges();
    }


    void setupEdges(){
        for (unsigned int i = 0; i < nodes.size(); i++){
            array<unsigned int, KNN_MAX+1> idx;
            findNearestNodes(back_up_nodes[i].Position, k_nearest + 1, idx);
            for (int j = 1; j < k_nearest+1; j++) {
                nodes[i].Neighbor[j-1] = idx[j]; 
                back_up_nodes[i].Neighbor[j-1] = idx[j]; 
                indices.push_back(i);
                indices.push_back(idx[j]);
            }
        }
    }

    void resetRT(){
        for (int j = 0; j < nodes.size(); j++){
            for(int i=0; i<9; i++) rot[j][i] = 0.0;
            rot[j][0] = 1.0; rot[j][4] = 1.0; rot[j][8] = 1.0;
            for(int i=0; i<3; i++) trans[j][i] = 0.0;
	    }
    }

    void resetGraph(){  
        vector<Node> back_up(this->back_up_nodes);
        this->nodes.clear();
        this->nodes = back_up;
        free_indices.clear();
        for (unsigned int i = 0; i < nodes.size(); i++){
            free_indices.insert(i);
        }
        resetRT();
        static_indices.clear();
        control_indices.clear();

        rot.clear();
        trans.clear();
        for (int j = 0; j < nodes.size(); j++){
            array<double, 9> rotation;
            array<double, 3> translation;
            for(int i=0; i<9; i++) rotation[i] = 0.0;
            rotation[0] = 1.0; rotation[4] = 1.0; rotation[8] = 1.0;
            for(int i=0; i<3; i++) translation[i] = 0.0;
            rot.push_back(rotation);
            trans.push_back(translation);
	    }

        setupEdges();
    }



    void moveControls(Pos delta_pos) {    //TODO: change to control the constraints
        for (unsigned int i : control_indices){
            nodes[i].Position = nodes[i].Position + delta_pos;
        }
    }

    void putFreeInputs(double* x, const vector<Pos>& vertices){
        int index = 0;
        for (std::set<unsigned int>::iterator it = free_indices.begin(); it != free_indices.end(); ++it) {
            for (int j = 0 ; j < 9; j++) {
                rot[*it][j] = x[index*12 + j];
            }
            for (int j = 0 ; j < 3; j++) {
                trans[*it][j] = x[index*12 + 9 + j];
            }
            index += 1;
        }
    }

    void findNearestNodes(const Pos pos, unsigned int k, array<unsigned int, KNN_MAX+1>& idx)const {
        //cout << nodes.size() << endl;
        double d[back_up_nodes.size()];
        double temp_d;
        unsigned int index[back_up_nodes.size()];
        unsigned int idx_min;
        int t;
        Pos temp;

        for(unsigned int j=0; j < back_up_nodes.size(); j++){
            index[j] = j;
            temp = pos - back_up_nodes[j].Position;
            d[j] = sqrt(temp(0)*temp(0)+temp(1)*temp(1)+temp(2)*temp(2));
        }
        for(unsigned int i=0; i<k; i++){   //selection sort
            idx_min = i;
            for(unsigned int j = back_up_nodes.size() - 1; j>i; j--) {
                if(d[j] < d[idx_min]) {
                    idx_min = j;
                }
            }
            idx[i] = index[idx_min];

            temp_d = d[i]; 
            t = index[i];

            d[i] = d[idx_min]; 
            index[i]=index[idx_min];

            d[idx_min] = temp_d; 
            index[idx_min] = t;
        }
    }

    void computeWeights(const Pos pos, std::array<double, KNN_MAX>& weights, std::array<unsigned int, KNN_MAX+1>& idx) const{

        double dist, sum;
        sum = 0.0;
        Pos temp;

        findNearestNodes(pos, k_nearest+1, idx);

        //compuate wj(v)
        temp = pos - back_up_nodes[idx[k_nearest]].Position;
        double dmax =  sqrt(temp(0)*temp(0)+temp(1)*temp(1)+temp(2)*temp(2));
        for(int j=0; j<k_nearest; j++){
            temp = pos - back_up_nodes[idx[j]].Position;
            dist = sqrt(temp(0)*temp(0)+temp(1)*temp(1)+temp(2)*temp(2));
            weights[j] = pow(1.0-dist/dmax, 2.0);
            // weights[j] = 1.0-dist/dmax;
            sum += weights[j];
        }
        //normalize to sum to 1
        if(k_nearest==1) weights[0]=1.0;
        else for(int j=0; j<k_nearest; j++) weights[j] /= sum;
    }


    Pos predict(const Vertex& v, const vector<Pos>& vertices) const{

        Pos temp;
        Pos output(0.0, 0.0, 0.0);
        Pos position; 

        for(int j=0; j<k_nearest; j++) {
            position = nodes[v.Neighbor_Nodes[j]].Position;
            temp = vertices[v.index] - position; // ??
            array<double, 9> rotation = rot[v.Neighbor_Nodes[j]];
            array<double, 3> translation = trans[v.Neighbor_Nodes[j]];
            output(0) += v.Neighbor_Weights[j]*(rotation[0]*temp(0) + rotation[3]*temp(1) + rotation[6]*temp(2) + translation[0] + position(0));
            output(1) += v.Neighbor_Weights[j]*(rotation[1]*temp(0) + rotation[4]*temp(1) + rotation[7]*temp(2) + translation[1] + position(1));
            output(2) += v.Neighbor_Weights[j]*(rotation[2]*temp(0) + rotation[5]*temp(1) + rotation[8]*temp(2) + translation[2] + position(2));
        }

        return output;
    }

    Pos predict_mesh(const Vertex& v, const vector<Pos>& vertices, Pos cur_mesh_v) const{

        Pos temp;
        Pos output(0.0, 0.0, 0.0);
        Pos position;

        //cout <<  v.Neighbor_Weights[0] << " " <<  v.Neighbor_Weights[1] << " " <<  v.Neighbor_Weights[2] << endl;    

        for(int j=0; j<k_nearest; j++) {
            position = nodes[v.Neighbor_Nodes[j]].Position;
            temp = cur_mesh_v - position; // ??
            array<double, 9> rotation = rot[v.Neighbor_Nodes[j]];
            array<double, 3> translation = trans[v.Neighbor_Nodes[j]];
            output(0) += v.Neighbor_Weights[j]*(rotation[0]*temp(0) + rotation[3]*temp(1) + rotation[6]*temp(2) + translation[0] + position(0));
            output(1) += v.Neighbor_Weights[j]*(rotation[1]*temp(0) + rotation[4]*temp(1) + rotation[7]*temp(2) + translation[1] + position(1));
            output(2) += v.Neighbor_Weights[j]*(rotation[2]*temp(0) + rotation[5]*temp(1) + rotation[8]*temp(2) + translation[2] + position(2));
        }

        return output;
    }

    Pos predict_samples(int idx, const SamplePoint& v, const vector<Pos>& vertices, const vector<Pos>& sample_positions) const{

        Pos temp;
        Pos output(0.0, 0.0, 0.0);
        Pos position;  

        for(int j=0; j<k_nearest; j++) {
            position = nodes[v.Neighbor_Nodes[j]].Position;
            temp = sample_positions[idx] - position; 
            array<double, 9> rotation = rot[v.Neighbor_Nodes[j]];
            array<double, 3> translation = trans[v.Neighbor_Nodes[j]];
            output(0) += v.Neighbor_Weights[j]*(rotation[0]*temp(0) + rotation[3]*temp(1) + rotation[6]*temp(2) + translation[0] + position(0));
            output(1) += v.Neighbor_Weights[j]*(rotation[1]*temp(0) + rotation[4]*temp(1) + rotation[7]*temp(2) + translation[1] + position(1));
            output(2) += v.Neighbor_Weights[j]*(rotation[2]*temp(0) + rotation[5]*temp(1) + rotation[8]*temp(2) + translation[2] + position(2));
        }

        return output;
    }

    Pos historical_predict_samples(int idx, const SamplePoint& v, const vector<Pos>& vertices, const vector<Pos>& sample_positions, int step){

        Pos temp;
        Pos output(0.0, 0.0, 0.0);
        Pos position;  

        for(int j=0; j<k_nearest; j++) {
            position = history_node_positions[step][v.Neighbor_Nodes[j]];
            temp = sample_positions[idx] - position; // ??
            array<double, 9> rotation = history_rot[step][v.Neighbor_Nodes[j]];
            array<double, 3> translation = history_trans[step][v.Neighbor_Nodes[j]];
            output(0) += v.Neighbor_Weights[j]*(rotation[0]*temp(0) + rotation[3]*temp(1) + rotation[6]*temp(2) + translation[0] + position(0));
            output(1) += v.Neighbor_Weights[j]*(rotation[1]*temp(0) + rotation[4]*temp(1) + rotation[7]*temp(2) + translation[1] + position(1));
            output(2) += v.Neighbor_Weights[j]*(rotation[2]*temp(0) + rotation[5]*temp(1) + rotation[8]*temp(2) + translation[2] + position(2));
        }

        return output;
    }

    Pos inversed_historical_predict_samples(int idx, const SamplePoint& v, const vector<Pos>& vertices, const vector<Pos>& sample_positions, int step) {

        Pos node_pos;

        float a = sample_positions[idx](0);
        float b = sample_positions[idx](1);
        float c = sample_positions[idx](2);
        array<float, 9> r = {0.0f};
        Pos output(0.0, 0.0, 0.0);

        for(int j=0; j<k_nearest; j++) {
            node_pos = history_node_positions[step][v.Neighbor_Nodes[j]];
            array<double, 9> rotation = history_rot[step][v.Neighbor_Nodes[j]];
            array<double, 3> translation = history_trans[step][v.Neighbor_Nodes[j]];

            for (int i = 0; i < 9; i++){
                r[i] += v.Neighbor_Weights[j]*(float)rotation[i];
            }

            a += v.Neighbor_Weights[j]*((float)rotation[0]*node_pos(0) + (float)rotation[3]*node_pos(1) + (float)rotation[6]*node_pos(2) - translation[0] - node_pos(0));
            b += v.Neighbor_Weights[j]*((float)rotation[1]*node_pos(0) + (float)rotation[4]*node_pos(1) + (float)rotation[7]*node_pos(2) - translation[1] - node_pos(1));
            c += v.Neighbor_Weights[j]*((float)rotation[2]*node_pos(0) + (float)rotation[5]*node_pos(1) + (float)rotation[8]*node_pos(2) - translation[2] - node_pos(2));
        }

        float divisor = (-r[0]*r[4]*r[8] + r[0]*r[5]*r[7] + r[1]*r[3]*r[8] - r[1]*r[5]*r[6] - r[2]*r[3]*r[7] + r[2]*r[4]*r[6]);
        output(0) = -(a*r[4]*r[8]-a*r[5]*r[7]-b*r[3]*r[8]+b*r[5]*r[6]+c*r[3]*r[7]-c*r[4]*r[6]) / divisor;
        output(1) = -(-a*r[1]*r[8]+a*r[2]*r[7]+b*r[0]*r[8]-b*r[2]*r[6]-c*r[0]*r[7]+c*r[1]*r[6]) / divisor;
        output(2) = -(-a*r[1]*r[5]+a*r[2]*r[4]+b*r[0]*r[5]-b*r[2]*r[3]-c*r[0]*r[4]+c*r[1]*r[3]) / (-divisor);

        return output;
    }


    Pos historical_predict(const Vertex& v, const vector<Pos>& vertices, int step) const{

        Pos temp;
        Pos output(0.0, 0.0, 0.0);
        Pos position;    

        for(int j=0; j<k_nearest; j++) {
            position = nodes[v.Neighbor_Nodes[j]].Position;
            temp = vertices[v.index] - position; // ??
            array<double, 9> rotation = history_rot[step][v.Neighbor_Nodes[j]];
            array<double, 3> translation = history_trans[step][v.Neighbor_Nodes[j]];
            output(0) += v.Neighbor_Weights[j]*(rotation[0]*temp(0) + rotation[3]*temp(1) + rotation[6]*temp(2) + translation[0] + position(0));
            output(1) += v.Neighbor_Weights[j]*(rotation[1]*temp(0) + rotation[4]*temp(1) + rotation[7]*temp(2) + translation[1] + position(1));
            output(2) += v.Neighbor_Weights[j]*(rotation[2]*temp(0) + rotation[5]*temp(1) + rotation[8]*temp(2) + translation[2] + position(2));
        }

        return output;
    }

    Pos inversed_historical_predict(const Vertex& v, const vector<Pos>& vertices, int step) {

        Pos node_pos;

        float a = vertices[v.index](0);
        float b = vertices[v.index](1);
        float c = vertices[v.index](2);
        array<float, 9> r = {0.0f};
        Pos output(0.0, 0.0, 0.0);

        for(int j=0; j<k_nearest; j++) {
            node_pos = history_node_positions[step][v.Neighbor_Nodes[j]];
            array<double, 9> rotation = history_rot[step][v.Neighbor_Nodes[j]];
            array<double, 3> translation = history_trans[step][v.Neighbor_Nodes[j]];

            for (int i = 0; i < 9; i++){
                r[i] += v.Neighbor_Weights[j]*(float)rotation[i];
            }

            a += v.Neighbor_Weights[j]*((float)rotation[0]*node_pos(0) + (float)rotation[3]*node_pos(1) + (float)rotation[6]*node_pos(2) - translation[0] - node_pos(0));
            b += v.Neighbor_Weights[j]*((float)rotation[1]*node_pos(0) + (float)rotation[4]*node_pos(1) + (float)rotation[7]*node_pos(2) - translation[1] - node_pos(1));
            c += v.Neighbor_Weights[j]*((float)rotation[2]*node_pos(0) + (float)rotation[5]*node_pos(1) + (float)rotation[8]*node_pos(2) - translation[2] - node_pos(2));
        }

        float divisor = (-r[0]*r[4]*r[8] + r[0]*r[5]*r[7] + r[1]*r[3]*r[8] - r[1]*r[5]*r[6] - r[2]*r[3]*r[7] + r[2]*r[4]*r[6]);
        output(0) = -(a*r[4]*r[8]-a*r[5]*r[7]-b*r[3]*r[8]+b*r[5]*r[6]+c*r[3]*r[7]-c*r[4]*r[6]) / divisor;
        output(1) = -(-a*r[1]*r[8]+a*r[2]*r[7]+b*r[0]*r[8]-b*r[2]*r[6]-c*r[0]*r[7]+c*r[1]*r[6]) / divisor;
        output(2) = -(-a*r[1]*r[5]+a*r[2]*r[4]+b*r[0]*r[5]-b*r[2]*r[3]-c*r[0]*r[4]+c*r[1]*r[3]) / (-divisor);

        return output;
    }

    
private:

};


class Deform{

public:

    set<unsigned int> *free_indices;
    set<unsigned int> *static_indices;
    vector<Node> *nodes;

    vector<Vertex> *vertices;
    std::vector<Pos> *pos_vector;


    vector<vector<unsigned int>> *indices_blocks;
    vector<int> *blocks_type;
    int control_blocks_num;
    int control_nodes_num;
    vector<Pos> aim_blocks_center;
    vector<set<unsigned int>> center_control_indices;


    vector<Pos> aim_positions;
    bool _constraints_on_center;

    double w_rotation;
    double w_regularization;
    double w_constraints;

    double control_weight_factor;

    int each_node_dim;
    bool _show_deform_efficiency;

    Vector1d m_x;
    int k_nearest;

    Deform(const std::vector<Vertex>& verts, const vector<Pos>& pos, const set<unsigned int>& control_idx, const set<unsigned int>& static_idx, const DeformGraph& deform_graph, vector<Pos> aim_pos, bool cons_on_center, bool _show_efficiency, int knn_k, const vector<int>& blocks_ts, float _weighting_factor):
        free_indices(&(deform_graph.free_indices)),
        static_indices(&(deform_graph.static_indices)),
        nodes(&(deform_graph.nodes)),
        vertices(&verts),
        pos_vector(&pos),
        w_rotation(deform_graph.w_rot),
        w_regularization(deform_graph.w_reg),
        w_constraints(deform_graph.w_con),
        aim_positions(aim_pos),
        _constraints_on_center(cons_on_center),
        _show_deform_efficiency(_show_efficiency),
        each_node_dim(12),
        k_nearest(knn_k),
        indices_blocks(&deform_graph.indices_blocks),
        blocks_type(&blocks_ts),
        control_weight_factor((double)_weighting_factor){
            
            
            GetControlNums();
            if (_constraints_on_center) {
                SelectKeyControls();
            }

            if (_constraints_on_center) {
                fx_n = (*free_indices).size() * (6 + 3 * k_nearest);
                fx_n += 3*control_blocks_num;
                fx_n += (*static_indices).size() * (3 * k_nearest);
                jacobi_n = each_node_dim * (*free_indices).size(); 
                jacobi_m = fx_n;
            }
            else{
                fx_n = (*free_indices).size() * (6 + 3 * k_nearest);
                fx_n += control_nodes_num * 3;
                fx_n += (*static_indices).size() * (3 * k_nearest);
                jacobi_n = each_node_dim * (*free_indices).size(); 
                jacobi_m = fx_n;
            }
            w_rotation = sqrt(w_rotation);
            w_regularization = sqrt(w_regularization);
            w_constraints = sqrt(w_constraints);
        }

    
    void real_time_deform();
    void setIdentityRots();
    double optimize();
    void DefineJacobiStructure(SpMat& jacobi, SpMat& jacobiT);
    void FastCalcJacobiMat(SpMat& jacobi, SpMat& jacobiT);
    void CalcEnergyFunc(Vector1d& fx, Vector1d& x);
    void SelectKeyControls();
    void GetControlNums();
    bool BelongsToSameBlock(unsigned int node_idx1, unsigned int node_idx2);

private:
    SpMat m_jacobi;
	SpMat m_jacobiT;
    int jacobi_m,jacobi_n,fx_n;
};


#endif