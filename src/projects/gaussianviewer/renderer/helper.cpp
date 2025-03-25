#include <helper.hpp>
#include <iostream>
#include <random>
#include <cmath>

#define kSqrt03_02    sqrt( 3.0 /  2.0)
#define kSqrt01_03    sqrt( 1.0 /  3.0)
#define kSqrt02_03    sqrt( 2.0 /  3.0)
#define kSqrt04_03    sqrt( 4.0 /  3.0)
#define kSqrt01_04    sqrt( 1.0 /  4.0)
#define kSqrt03_04    sqrt( 3.0 /  4.0)
#define kSqrt01_05    sqrt( 1.0 /  5.0)
#define kSqrt03_05    sqrt( 3.0 /  5.0)
#define kSqrt06_05    sqrt( 6.0 /  5.0)
#define kSqrt08_05    sqrt( 8.0 /  5.0)
#define kSqrt09_05    sqrt( 9.0 /  5.0)
#define kSqrt05_06    sqrt( 5.0 /  6.0)
#define kSqrt01_06    sqrt( 1.0 /  6.0)
#define kSqrt03_08    sqrt( 3.0 /  8.0)
#define kSqrt05_08    sqrt( 5.0 /  8.0)
#define kSqrt07_08    sqrt( 7.0 /  8.0)
#define kSqrt09_08    sqrt( 9.0 /  8.0)
#define kSqrt05_09    sqrt( 5.0 /  9.0)
#define kSqrt08_09    sqrt( 8.0 /  9.0)

#define kSqrt01_10    sqrt( 1.0 / 10.0)
#define kSqrt03_10    sqrt( 3.0 / 10.0)
#define kSqrt01_12    sqrt( 1.0 / 12.0)
#define kSqrt04_15    sqrt( 4.0 / 15.0)
#define kSqrt01_16    sqrt( 1.0 / 16.0)
#define kSqrt07_16    sqrt( 7.0 / 16.0)
#define kSqrt15_16    sqrt(15.0 / 16.0)
#define kSqrt01_18    sqrt( 1.0 / 18.0)
#define kSqrt03_25    sqrt( 3.0 / 25.0)
#define kSqrt14_25    sqrt(14.0 / 25.0)
#define kSqrt15_25    sqrt(15.0 / 25.0)
#define kSqrt18_25    sqrt(18.0 / 25.0)
#define kSqrt01_32    sqrt( 1.0 / 32.0)
#define kSqrt03_32    sqrt( 3.0 / 32.0)
#define kSqrt15_32    sqrt(15.0 / 32.0)
#define kSqrt21_32    sqrt(21.0 / 32.0)
#define kSqrt01_50    sqrt( 1.0 / 50.0)
#define kSqrt03_50    sqrt( 3.0 / 50.0)
#define kSqrt21_50    sqrt(21.0 / 50.0)
#define kSqrt1_60    sqrt(1.0 / 60.0)

#define Infinity    100000.0



sibr::Vector2f get_2d_pos(Pos pos_3d, sibr::Matrix4f T, const unsigned int width, const unsigned int height)
{
    sibr::Vector4f gl_pos = T * sibr::Vector4f(pos_3d(0), pos_3d(1), pos_3d(2), 1.0);
    double screen_x = (gl_pos(0) / gl_pos(3) + 1) * width / 2;
    double screen_y = (-gl_pos(1) / gl_pos(3) + 1) * height / 2;
    // std::cout << "screen_x" << screen_x << "screen_y" << screen_y << std::endl;
    return sibr::Vector2f(screen_x, screen_y);
}

float pts_distance(const Pos pos1, const Pos pos2)
{
    return std::sqrt(std::pow(pos1(0) - pos2(0), 2) + std::pow(pos1(1) - pos2(1), 2) + std::pow(pos1(2) - pos2(2), 2));
}

// a naive implementation without acceleration
void random_sample_once(const std::vector<Pos> &vertices, std::vector<int> &node_idxs, float threshold)
{
    std::vector<int> sample_idxs;
    std::random_device rd;
    std::mt19937 gen(rd());

    while (!node_idxs.empty())
    {
        // std::cout << "Deleting Points..." << "Points Left:" << node_idxs.size() << std::endl;
        std::uniform_int_distribution<> dis(0, node_idxs.size() - 1);
        int idx = dis(gen);
        Pos sampled_node = vertices[node_idxs[idx]];
        sample_idxs.push_back(node_idxs[idx]);
        node_idxs.erase(node_idxs.begin() + idx);

        for (auto it = node_idxs.begin(); it != node_idxs.end();)
        {
            Pos vertex = vertices[*it];
            float dist = pts_distance(vertex, sampled_node);
            if (dist < threshold)
            {
                it = node_idxs.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }
    node_idxs = sample_idxs;

    // std::cout << "Size of Sample_idxs" << sample_idxs.size() << std::endl;
    // std::cout << "Size of Node_idx" << node_idxs.size() << std::endl;
}

std::vector<int> sample_control_points(const std::vector<Pos> &vertices, int node_num, float threshold_d, bool only_surface, const std::vector<int> &is_surface_vector)
{
    std::vector<int> node_idx(vertices.size());
    std::iota(node_idx.begin(), node_idx.end(), 0);

    if (only_surface){
        int idx = 0;
        for (auto it = node_idx.begin(); it != node_idx.end();){
            if ((abs(vertices[idx](0)) < 0.27) && (abs(vertices[idx](2)) < 0.27))
            {
                it = node_idx.erase(it);
            }
            else
            {
                ++it;
            }
            idx++;
        }
    }

    std::vector<int> node_idx_backup;
    node_idx_backup = node_idx;

    std::cout << "at beginning, number of nodes " << node_idx.size() << std::endl;

    while (node_idx.size() > node_num)
    {
        node_idx = node_idx_backup;
        std::cout << "threshold " << threshold_d << std::endl;
        random_sample_once(vertices, node_idx, threshold_d);
        threshold_d = threshold_d * 1.2;
        std::cout << "currently, number of nodes " << node_idx.size() << std::endl;
    }

    return node_idx;
}


std::vector<int> farthest_control_points_sampling(const std::vector<Pos> pos_vector, int node_num, bool only_surface){

    std::vector<unsigned int> origin_indices;

    int node_num_cur = node_num;
    if (node_num_cur > pos_vector.size()){
        node_num_cur = pos_vector.size();
    }

    std::vector<int> node_idx;
    node_idx.reserve(node_num_cur);

    int first_node_idx;
    first_node_idx = FetchFirstNodeIdx(pos_vector);

    node_idx.push_back(first_node_idx);

    std::vector<float> distances(pos_vector.size(), std::numeric_limits<float>::max());

    for (int i = 0; i < node_num_cur; i++){
        if (first_node_idx == i) {continue;}
        Pos lastsamplePoint = pos_vector[node_idx.back()];
        for (int j = 0; j < pos_vector.size(); j++){
            float distance = pts_distance(lastsamplePoint, pos_vector[j]);
            if (distance < distances[j]) {
                distances[j] = distance;
            }
        }
        auto maxElementIter = std::max_element(distances.begin(), distances.end());
        size_t farthestPointIdx = std::distance(distances.begin(), maxElementIter);
        node_idx.push_back(farthestPointIdx);

        if (node_idx.size() == node_num_cur) {
            break;
        }
    }
    
    return node_idx;
}


int FetchFirstNodeIdx(vector<Pos> node_postions){
    int first_node_idx = 0;
    float distance = (float)-Infinity;

    for (int i = 0; i < node_postions.size(); i++){
        float cur_dist = 0.0f;
        for (int j = 0; j < 3; j++) {
            cur_dist += node_postions[i](j);
        }
        if (cur_dist > distance){
            distance = cur_dist;
            first_node_idx = i;
        }
    }
    return first_node_idx;
}


void storeVectorToFile(const std::vector<int>& data, const std::string& filename) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        for (int i : data) {
            outFile << i << " ";
        }
        outFile.close();
        std::cout << "Vector已存储到" << filename << "文件中" << std::endl;
    } else {
        std::cerr << "无法打开文件" << filename << std::endl;
    }
}

std::vector<int> loadVectorFromFile(const std::string& filename) {
    std::ifstream inFile(filename);
    std::vector<int> data;
    if (inFile.is_open()) {
        std::string line;
        while (std::getline(inFile, line)) {
            std::istringstream iss(line);
            int value;
            while (iss >> value) {
                //std::cout << "node_idx: " << value << std::endl;
                data.push_back(value);
            }
        }
        inFile.close();
        std::cout << "从" << filename << "文件中读取Vector成功" << std::endl;
    } else {
        std::cerr << "无法打开文件" << filename << std::endl;
    }
    return data;
}

void readObjFile(const std::string& filename, std::vector<Vertex_m>& vertices, std::vector<Normal_m>& normals, std::vector<Face_m>& faces) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.substr(0, 2) == "v ") {
            std::istringstream ss(line.substr(2));
            Vertex_m vertex;
            ss >> vertex.x >> vertex.y >> vertex.z;
            vertices.push_back(vertex);
        } else if (line.substr(0, 3) == "vn ") {
            std::istringstream ss(line.substr(3));
            Normal_m normal;
            ss >> normal.nx >> normal.ny >> normal.nz;
            normals.push_back(normal);
        } else if (line.substr(0, 2) == "f ") {
            std::istringstream ss(line.substr(2));
            Face_m face;
            char separator;
            ss >> face.v1 >> separator >> separator >> face.vn1 >> face.v2 >> separator >> separator >> face.vn2 >> face.v3 >> separator >> separator >> face.vn3;
            faces.push_back(face);
        }
    }

    file.close();
}


void LoadMeshPoints(const std::string& filename, std::vector<Pos> &meshPoints){
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.substr(0, 2) == "v ") {
            std::istringstream ss(line.substr(2));
            Pos Points;
            ss >> Points(0) >> Points(1) >> Points(2);
            meshPoints.push_back(Points);
        }
    }

    file.close();
}

void saveVectorToFile(const std::vector<float>& values, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {
        for (const float& value : values) {
            outputFile << value << std::endl;
        }
        outputFile.close();
        std::cout << "Vector saved to " << filename << " successfully." << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}

void saveVectorToFile(const std::vector<SHs<3>>& values, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {
        for(SHs<3> Sh : values) {
            for (int i = 0; i < 3; i++) {
                outputFile << Sh.shs[i] << std::endl;
            }
        }
        outputFile.close();
        std::cout << "Vector saved to " << filename << " successfully." << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}

void saveVectorToFile(const std::vector<Scale>& values, const std::string& filename){
    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {
        for(Scale s : values) {
            for (int i = 0; i < 3; i++) {
                outputFile << s.scale[i] << std::endl;
            }
        }
        outputFile.close();
        std::cout << "Vector saved to " << filename << " successfully." << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}

void saveVectorToFile(const std::vector<Pos>& values, const std::string& filename){
    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {
        for(Pos p : values) {
            outputFile << p(0) << " " << p(1) << " " << p(2) << std::endl;
        }
        outputFile.close();
        std::cout << "Vector saved to " << filename << " successfully." << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}

void saveVectorToFile(const std::vector<Rot>& values, const std::string& filename){
    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {
        for(Rot r : values) {
            for (int i = 0; i < 4; i++) {
                outputFile << r.rot[i] << std::endl;
            }
        }
        outputFile.close();
        std::cout << "Vector saved to " << filename << " successfully." << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}


void saveEigenMatrixVectorToFile(const std::vector<Eigen::Matrix3f>& matrixVector, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {
        for (const Eigen::Matrix3f& matrix : matrixVector) {
            outputFile << matrix << std::endl << std::endl;
        }
        outputFile.close();
        std::cout << "Matrix vector saved to " << filename << " successfully." << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}



void saveEigenVectorToTxt(const std::vector<Pos>& vec, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (const Pos& v : vec) {
            file << v.x() << " " << v.y() << " " << v.z() << std::endl;
        }
        file.close();
        std::cout << "Vector saved to " << filename << " successfully." << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}

std::vector<float> readFloatsFromFile(const std::string& filename) {
    std::vector<float> floatVector;
    std::ifstream inputFile(filename);

    if (inputFile.is_open()) {
        std::string line;
        while (std::getline(inputFile, line)) {
            std::istringstream iss(line);
            float value;
            while (iss >> value) {
                floatVector.push_back(value);
            }
        }
        inputFile.close();
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }

    return floatVector;
}

void polarDecomposition(const Eigen::Matrix3f A, Eigen::Matrix3f& S){
    Eigen::Matrix3f ATA = A.transpose()*A;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(ATA);
    Eigen::Matrix3f eigenvectors = eigenSolver.eigenvectors();
    Eigen::Matrix3f eigenvalues = eigenSolver.eigenvalues().asDiagonal();

    S = eigenvectors * eigenvalues.cwiseSqrt() * eigenvectors.transpose();
}

Eigen::Matrix3f getOthogonalMatrix(Eigen::Matrix3f mat){
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat.cast<double>(), Eigen::ComputeThinU | Eigen::ComputeThinV);
    return (svd.matrixU() * svd.matrixV().transpose()).cast<float>();

    // Eigen::HouseholderQR<Eigen::MatrixXf> qr(mat);
    // return qr.householderQ();
}

Eigen::Vector3f getSingularValues(Eigen::Matrix3f mat){
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

    return svd.singularValues();
}

Eigen::Matrix3f getOthogonalMatrixWithK(Eigen::Matrix3f mat,Eigen::Vector3f& K){
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat.cast<double>(), Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3f S = (V * svd.singularValues().asDiagonal()*V.transpose()).cast<float>();
    Eigen::Matrix3f R = (U*V.transpose()).cast<float>();

    K = S.diagonal();

    return R;
}


Eigen::Matrix3f getOthogonalMatrixWithCheck(Eigen::Matrix3f mat){
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat.cast<double>(), Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3f S = (V * svd.singularValues().asDiagonal()*V.transpose()).cast<float>();
    Eigen::Matrix3f R = (U*V.transpose()).cast<float>();

    float reconstruct_error = (mat - R*S).norm();

    if (reconstruct_error > 1e-3){
        std::cout << "RS reconstruct error: " << reconstruct_error << std::endl;
        std::cout << "mat: " << mat << std::endl;
        std::cout << "recon: " << R*S << std::endl;
    }
    
    return R;
}

bool CheckReconstructionError(Eigen::Matrix3f mat, Eigen::Matrix3f R, Eigen::Vector3f K){
    Eigen::Matrix3f S;
    S.setZero();
    S.diagonal() = K;

    float reconstruct_error = (mat - R*S).norm();

    if (reconstruct_error > 1e-3){
        std::cout << "RK reconstruct error: " << reconstruct_error << std::endl;
        std::cout << "mat: " << mat << std::endl;
        std::cout << "recon: " << R*S << std::endl;
        return false;
    }
    return true;
}

bool CheckRotationContinue(Eigen::Quaternionf q0, Eigen::Quaternionf last_q){
    Eigen::Matrix3f R0 = q0.toRotationMatrix();
    Eigen::Matrix3f last_R = last_q.toRotationMatrix();

    Eigen::Matrix3f R = R0.transpose() * last_R;
    float trace_R = R.trace();
    float angle = std::acos((trace_R - 1.0f) / 2.0f);

    if (angle > Pi / 4.0f){
        std::cout << "angle" << angle << std::endl;
        return false;
    }
    return true;
}

bool CheckRotation(Eigen::Matrix3f R0, Eigen::Matrix3f R1){

    Eigen::Matrix3f R = R0.transpose() * R1;
    float trace_R = R.trace();
    float angle = std::acos((trace_R - 1.0f) / 2.0f);

    if (angle > Pi / 100.0f){
        std::cout << "angle" << angle << std::endl;
        return false;
    }
    return true;
}

Eigen::Matrix3f FastgetOthogonalMatrix(Eigen::Matrix3f mat){
    Eigen::Matrix3f M = mat;
    Eigen::Matrix3f next_mat;
    Eigen::Matrix3f mSwap;
    while(true){
        mSwap = M.transpose();
        next_mat = 0.5*(M + mSwap.inverse());
        if ((M - next_mat).cwiseAbs().maxCoeff() < 1e-6){ break;}
        M = next_mat;
    }
    return M;
}

Eigen::Matrix3f FastgetOthogonalMatrixWithK(Eigen::Matrix3f mat, Eigen::Vector3f& K){
    Eigen::Matrix3f M = mat;
    Eigen::Matrix3f next_mat;
    Eigen::Matrix3f mSwap;
    while(true){
        mSwap = M.transpose();
        next_mat = 0.5*(M + mSwap.inverse()); //TODO if not inversible
        if ((M - next_mat).cwiseAbs().maxCoeff() < 1e-6){ break;}
        M = next_mat;
    }
    Eigen::Matrix3f S = M.transpose() * mat;
    K = S.diagonal();
    return M;
}

Eigen::Matrix3f FastgetK(Eigen::Matrix3f Affine, Eigen::Matrix3f R, Eigen::Vector3f& K, Eigen::Matrix3f R0){
    Eigen::Matrix3f S = R.transpose() * Affine;
    // S = 0.5*(S.transpose() + S);
    K = GetKfromSR(S, R0);
    return S;
}

Eigen::Matrix3f FastgetOthogonalMatrixAndK(Eigen::Matrix3f mat, Eigen::Vector3f& K, Eigen::Matrix3f R0){
    Eigen::Matrix3f M = mat;
    Eigen::Matrix3f next_mat = mat;
    Eigen::Matrix3f mSwap = mat.transpose();

    while(true){
        mSwap = mat.transpose();
        next_mat = 0.5*(mat + mSwap.inverse());
        if ((mat - next_mat).cwiseAbs().maxCoeff() < 0.00001){ break;}
        mat = next_mat;
    }
    Eigen::Matrix3f R = mat;
    Eigen::Matrix3f S = R.transpose() * M;
    S = 0.5*(S.transpose() + S);
 
    K = GetKfromSR(S, R0);

    return R;
}

Eigen::Vector3f GetKfromSR(Eigen::Matrix3f S, Eigen::Matrix3f R){
    Eigen::Matrix<float, 9, 3> A;
    A <<    R.row(0)(0)*R.row(0)(0), R.row(0)(1)*R.row(0)(1), R.row(0)(2)*R.row(0)(2),
            R.row(0)(0)*R.row(1)(0), R.row(0)(1)*R.row(1)(1), R.row(0)(2)*R.row(1)(2),
            R.row(0)(0)*R.row(2)(0), R.row(0)(1)*R.row(2)(1), R.row(0)(2)*R.row(2)(2),
            R.row(1)(0)*R.row(0)(0), R.row(1)(1)*R.row(0)(1), R.row(1)(2)*R.row(0)(2),
            R.row(1)(0)*R.row(1)(0), R.row(1)(1)*R.row(1)(1), R.row(1)(2)*R.row(1)(2),
            R.row(1)(0)*R.row(2)(0), R.row(1)(1)*R.row(2)(1), R.row(1)(2)*R.row(2)(2),
            R.row(2)(0)*R.row(0)(0), R.row(2)(1)*R.row(0)(1), R.row(2)(2)*R.row(0)(2),
            R.row(2)(0)*R.row(1)(0), R.row(2)(1)*R.row(1)(1), R.row(2)(2)*R.row(1)(2),
            R.row(2)(0)*R.row(2)(0), R.row(2)(1)*R.row(2)(1), R.row(2)(2)*R.row(2)(2); 
    
    Eigen::VectorXf b(9);
    b << S.row(0)(0), S.row(0)(1), S.row(0)(2), 
        S.row(1)(0), S.row(1)(1), S.row(1)(2), 
        S.row(2)(0), S.row(2)(1), S.row(2)(2);
    
    Eigen::Vector3f K = A.colPivHouseholderQr().solve(b);

    // Eigen::MatrixXf AtA = A.transpose()*A;
	// Eigen::Matrix<float, 3, 1> Atb = A.transpose() * b;
    // Eigen::Vector3f K = (AtA).ldlt().solve(Atb);
    K = K.cwiseAbs();
    return K;
}


// void chooseDevice()
// {
//     torch::manual_seed(1);
//     torch::DeviceType device_type;
//     if (torch::cuda::is_available()) {
//         std::cout << "CUDA available! Training on GPU." << std::endl;
//         device_type = torch::kCUDA;
//     } 
//     else {
//         std::cout << "Training on CPU." << std::endl;
//         device_type = torch::kCPU;
//     }
//     torch::Device device(device_type);
// }
































Eigen::Vector3f spectDecomp(Eigen::Matrix3f S, Eigen::Matrix3f& U)
    {
        Eigen::Vector3f kv;
        double Diag[3],OffD[3]; /* OffD is off-diag (by omitted index) */
        double g,h,fabsh,fabsOffDi,t,theta,c,s,tau,ta,OffDq,a,b;
        static int nxt[] = {1,2,0};
        int sweep;
        U = Eigen::Matrix3f::Identity();
        Diag[0] = S.row(0)(0); Diag[1] = S.row(1)(1); Diag[2] = S.row(2)(2);
        OffD[0] = S.row(1)(2); OffD[1] = S.row(2)(0); OffD[2] = S.row(0)(1);
        for (sweep=20; sweep>0; sweep--) {
            double sm = fabs(OffD[0])+fabs(OffD[1])+fabs(OffD[2]);
            if (sm==0.0) break;
            for (int i=2; i>=0; i--) {
                int p = nxt[i]; int q = nxt[p];
                fabsOffDi = fabs(OffD[i]);
                g = 100.0*fabsOffDi;
                if (fabsOffDi>0.0) {
                    h = Diag[q] - Diag[p];
                    fabsh = fabs(h);
                    if (fabsh+g==fabsh) {
                        t = OffD[i]/h;
                    } else {
                        theta = 0.5*h/OffD[i];
                        t = 1.0/(fabs(theta)+sqrt(theta*theta+1.0));
                        if (theta<0.0) t = -t;
                    }
                    c = 1.0/sqrt(t*t+1.0); s = t*c;
                    tau = s/(c+1.0);
                    ta = t*OffD[i]; OffD[i] = 0.0;
                    Diag[p] -= ta; Diag[q] += ta;
                    OffDq = OffD[q];
                    OffD[q] -= s*(OffD[p] + tau*OffD[q]);
                    OffD[p] += s*(OffDq   - tau*OffD[p]);
                    for (int j=2; j>=0; j--) {
                        a = U.row(j)(p); b = U.row(j)(q);
                        U.row(j)(p) -= s*(b + tau*a);
                        U.row(j)(q) += s*(a - tau*b);
                    }
                }
            }
        }
        kv(0) = Diag[0]; kv(1) = Diag[1]; kv(2) = Diag[2];
        return (kv);
    }


Eigen::Quaternionf snuggle(Eigen::Quaternionf q, Eigen::Vector3f& k)
    {
#define sgn(n,v)    ((n)?-(v):(v))
#define swap(a,i,j) {a[3]=a[i]; a[i]=a[j]; a[j]=a[3];}
#define cycle(a,p)  if (p) {a[3]=a[0]; a[0]=a[1]; a[1]=a[2]; a[2]=a[3];}\
        else   {a[3]=a[2]; a[2]=a[1]; a[1]=a[0]; a[0]=a[3];}

        
        Eigen::Quaternionf p; p.w()=1.0; p.x()=0.0; p.y()=0.0; p.z()=0.0;
        double ka[4];
        int turn = -1;
        ka[0] = k(0); ka[1] = k(1); ka[2] = k(2);

        if (ka[0]==ka[1]) {
            if (ka[0]==ka[2])
                turn = 3;
            else turn = 2;
        }
        else {
            if (ka[0]==ka[2])
                turn = 1;
            else if (ka[1]==ka[2])
                turn = 0;
        }
        if (turn>=0) {
            Eigen::Quaternionf qxtoz, qytoz, qp, qtoz;
            qxtoz.w()= sqrt(0.5); qxtoz.y()= sqrt(0.5); qxtoz.x()= 0.0; qxtoz.z()=0.0;
            qytoz.w()= sqrt(0.5); qytoz.x()= sqrt(0.5); qytoz.y()= 0.0; qytoz.z()=0.0;
            unsigned int  win;
            double mag[3], t;
            switch (turn) {
                default: return (q.conjugate());
                case 0: q = q*qxtoz; swap(ka,0,2) break;
                case 1: q = q*qytoz; swap(ka,1,2) break;
                case 2: qtoz.w()=1.0; qtoz.x()=0.0; qtoz.y()=0.0; qtoz.z()=0.0; break;
            }
            q = q.conjugate();
            mag[0] = (double)q.z()*q.z()+(double)q.w()*q.w()-0.5;
            mag[1] = (double)q.x()*q.z()-(double)q.y()*q.w();
            mag[2] = (double)q.y()*q.z()+(double)q.x()*q.w();

            bool neg[3];
            for (int i=0; i<3; i++)
            {
                neg[i] = (mag[i]<0.0);
                if (neg[i]) mag[i] = -mag[i];
            }

            if (mag[0]>mag[1]) {
                if (mag[0]>mag[2])
                    win = 0;
                else win = 2;
            }
            else {
                if (mag[1]>mag[2]) win = 1;
                else win = 2;
            }

            switch (win) {
                case 0: if (neg[0]) p = Eigen::Quaternionf(Eigen::Vector4f(1.0, 0.0, 0.0, 0.0)); else p = Eigen::Quaternionf(Eigen::Vector4f(0.0, 0.0, 0.0, 1.0)); break;
                case 1: if (neg[1]) p = Eigen::Quaternionf(Eigen::Vector4f(0.5, 0.5, -0.5, -0.5)); else p = Eigen::Quaternionf(Eigen::Vector4f(0.5, 0.5, 0.5, 0.5));; cycle(ka,0) break;
                case 2: if (neg[2]) p = Eigen::Quaternionf(Eigen::Vector4f(-0.5, 0.5, -0.5, -0.5)); else p = Eigen::Quaternionf(Eigen::Vector4f(0.5, 0.5, 0.5, -0.5)); cycle(ka,1) break;
            }

            qp = q * p;
            t = sqrt(mag[win]+0.5);
            p = p * Eigen::Quaternionf(Eigen::Vector4f(0.0,0.0,-qp.z()/t,qp.w()/t));
            p = qtoz * p.conjugate();
        }
        else {
            double qa[4], pa[4];
            unsigned int lo, hi;
            bool par = false;
            bool neg[4];
            double all, big, two;
            qa[0] = q.x(); qa[1] = q.y(); qa[2] = q.z(); qa[3] = q.w();
            for (int i=0; i<4; i++) {
                pa[i] = 0.0;
                neg[i] = (qa[i]<0.0);
                if (neg[i]) qa[i] = -qa[i];
                par ^= neg[i];
            }

            /* Find two largest components, indices in hi and lo */
            if (qa[0]>qa[1]) lo = 0;
            else lo = 1;

            if (qa[2]>qa[3]) hi = 2;
            else hi = 3;

            if (qa[lo]>qa[hi]) {
                if (qa[lo^1]>qa[hi]) {
                    hi = lo; lo ^= 1;
                }
                else {
                    hi ^= lo; lo ^= hi; hi ^= lo;
                }
            }
            else {
                if (qa[hi^1]>qa[lo]) lo = hi^1;
            }

            all = (qa[0]+qa[1]+qa[2]+qa[3])*0.5;
            two = (qa[hi]+qa[lo])*sqrt(0.5);
            big = qa[hi];
            if (all>two) {
                if (all>big) {/*all*/
                    {for (int i=0; i<4; i++) pa[i] = sgn(neg[i], 0.5);}
                    cycle(ka,par);
                }
                else {/*big*/ pa[hi] = sgn(neg[hi],1.0);}
            } else {
                if (two>big) { /*two*/
                    pa[hi] = sgn(neg[hi],sqrt(0.5));
                    pa[lo] = sgn(neg[lo], sqrt(0.5));
                    if (lo>hi) {
                        hi ^= lo; lo ^= hi; hi ^= lo;
                    }
                    if (hi==3) {
                        hi = "\001\002\000"[lo];
                        lo = 3-hi-lo;
                    }
                    swap(ka,hi,lo);
                }
                else {/*big*/
                    pa[hi] = sgn(neg[hi],1.0);
                }
            }
            p.x() = -pa[0]; p.y() = -pa[1]; p.z() = -pa[2]; p.w() = pa[3];
        }
        k(0) = ka[0]; k(1) = ka[1]; k(2) = ka[2];
        return (p);
    }


Eigen::Matrix3f CastRot2Matrix(std::array<double, 9> rot){
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> matrix(rot.data()); 
    Eigen::Matrix3f mat = matrix.cast<float>(); 
    return mat;
}


Eigen::Quaternionf Quaternion_S_lerp(Eigen::Quaternionf const &start_q, Eigen::Quaternionf &end_q, double t)
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

void RunTestQ(){

}

void RunTestVKVt(Scale& s){
    Eigen::Matrix3f mat, R0, R, X;
    Eigen::Vector3f K;
    float error;

    std::cout << "Test 0" << std::endl;
    mat <<  100, 0, 0,
            0, 5, 0,
            0, 0, 1;
    R0  <<  1, 0, 0,
            0, 0, 1,
            0, -1, 0;
    R = FastgetOthogonalMatrixAndK(mat, K, R0);
    X.setZero();
    X.diagonal() = K;
    std::cout << "mat: \n" << mat << std::endl;
    error = (mat - R*(R0*(X*R0.transpose()))).cwiseAbs().maxCoeff();
    std::cout << "Error of Test 0: " << error << std::endl;

    s.scale[0] = s.scale[0] * K(0);
    s.scale[1] = s.scale[1] * K(1);
    s.scale[2] = s.scale[2] * K(2);


    std::cout << "Test 1" << std::endl;
    mat <<  100, 0.1, 0,
            0, 5, 0,
            0, 0, 1;
    R0  <<  1, 0, 0,
            0, 0, 1,
            0, -1, 0;
    R = FastgetOthogonalMatrixAndK(mat, K, R0);
    X.setZero();
    X.diagonal() = K;
    std::cout << "mat: \n" << mat << std::endl;
    error = (mat - R*(R0*(X*R0.transpose()))).cwiseAbs().maxCoeff();
    std::cout << "Error of Test 1: " << error << std::endl;

    s.scale[0] = s.scale[0] * K(0);
    s.scale[1] = s.scale[1] * K(1);
    s.scale[2] = s.scale[2] * K(2);



    std::cout << "Test 5" << std::endl;
    mat <<  9, -7, 2,
            -6, 3, 4,
            5, 1, -8;
    R0  <<  0, 0, 1,
            1, 0, 0,
            0, -1, 0;
    R = FastgetOthogonalMatrixAndK(mat, K, R0);
    X.setZero();
    X.diagonal() = K;
    std::cout << "mat: \n" << mat << std::endl;
    error = (mat - R*(R0*(X*R0.transpose()))).cwiseAbs().maxCoeff();
    std::cout << "Error of Test 5: " << error << std::endl;

}

void Test_SH_Rotation(){
    Eigen::Quaternionf origin_q(0.992198, -2.06755e-09, 0.124675, -1.99438e-09);
    Eigen::Matrix3f rots = origin_q.normalized().toRotationMatrix();

    std::vector<float> shs_in = {-1.27693, -1.2788, 1.59567, 0.420816, 0.424887, 0.159113, -0.392291, -0.394426, -0.00353546, -0.134855, -0.134582, 0.279297, -0.0708386, -0.0703888, -0.122712, 0.0842594, 0.086441, 0.0141932, -0.0287304, -0.0303147, 0.170433, -0.225444, -0.228179, 0.0991045, -0.489956, -0.495507, -0.128272, -0.0521032, -0.0514518, 0.0130957, -0.228547, -0.232625, -0.0120403, -0.197404, -0.197122, -0.0119539, -0.072907, -0.0705575, 0.0106995, 0.055612, 0.0594086, -0.0702121, 0.115523, 0.11532, -0.108263, 0.0875333, 0.0893422, 0.0194876};

    SH_Rotation(rots, shs_in);

    std::cout << "output: " << std::endl;
    for (int k = 0; k < 48; k++) {
        std::cout << shs_in[k] << ", ";	
    }
}

void SH_Rotation(Eigen::Matrix3f rots, std::vector<float>& Shs){
    Eigen::Matrix<float, 3, 3> sh1;
    Eigen::Matrix<float, 5, 5> sh2;
    Eigen::Matrix<float, 7, 7> sh3;
    Construct_SH_Rotation_Matrix(sh1, sh2, sh3, rots);

    Eigen::VectorXf r(16);
    Eigen::VectorXf g(16);
    Eigen::VectorXf b(16);

    for (int i = 0; i < 16; i++){
        r(i) = Shs[i*3];
        g(i) = Shs[i*3 + 1];
        b(i) = Shs[i*3 + 2];
    }

    r.segment(1, 3) = sh1*r.segment(1, 3);
    g.segment(1, 3) = sh1*g.segment(1, 3);
    b.segment(1, 3) = sh1*b.segment(1, 3);

    r.segment(4, 5) = sh2*r.segment(4, 5);
    g.segment(4, 5) = sh2*g.segment(4, 5);
    b.segment(4, 5) = sh2*b.segment(4, 5);

    r.segment(9, 7) = sh3*r.segment(9, 7);
    g.segment(9, 7) = sh3*g.segment(9, 7);
    b.segment(9, 7) = sh3*b.segment(9, 7);

    for (int i = 0; i < 16; i++){
        Shs[i*3] = r(i);
        Shs[i*3 + 1] = g(i);
        Shs[i*3 + 2] = b(i);
    }

    return; 
}

void Construct_SH_Rotation_Matrix(Eigen::Matrix<float, 3, 3>& sh1, 
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

    sh2.row(0)(0) = kSqrt01_04 * ((sh1.row(2)(2) * sh1.row(0)(0) + sh1.row(2)(0) * sh1.row(0)(2)) + (sh1.row(0)(2) * sh1.row(2)(0) + sh1.row(0)(0) * sh1.row(2)(2)));
    sh2.row(0)(1) = (sh1.row(2)(1) * sh1.row(0)(0) + sh1.row(0)(1) * sh1.row(2)(0));
    sh2.row(0)(2) = kSqrt03_04 * (sh1.row(2)(1) * sh1.row(0)(1) + sh1.row(0)(1) * sh1.row(2)(1));
    sh2.row(0)(3) = (sh1.row(2)(1) * sh1.row(0)(2) + sh1.row(0)(1) * sh1.row(2)(2));
    sh2.row(0)(4) = kSqrt01_04 * ((sh1.row(2)(2) * sh1.row(0)(2) - sh1.row(2)(0) * sh1.row(0)(0)) + (sh1.row(0)(2) * sh1.row(2)(2) - sh1.row(0)(0) * sh1.row(2)(0)));

    sh2.row(1)(0) = kSqrt01_04 * ((sh1.row(1)(2) * sh1.row(0)(0) + sh1.row(1)(0) * sh1.row(0)(2)) + (sh1.row(0)(2) * sh1.row(1)(0) + sh1.row(0)(0) * sh1.row(1)(2)));
    sh2.row(1)(1) = sh1.row(1)(1) * sh1.row(0)(0) + sh1.row(0)(1) * sh1.row(1)(0);
    sh2.row(1)(2) = kSqrt03_04 * (sh1.row(1)(1) * sh1.row(0)(1) + sh1.row(0)(1) * sh1.row(1)(1));
    sh2.row(1)(3) = sh1.row(1)(1) * sh1.row(0)(2) + sh1.row(0)(1) * sh1.row(1)(2);
    sh2.row(1)(4) = kSqrt01_04 * ((sh1.row(1)(2) * sh1.row(0)(2) - sh1.row(1)(0) * sh1.row(0)(0)) + (sh1.row(0)(2) * sh1.row(1)(2) - sh1.row(0)(0) * sh1.row(1)(0)));

    sh2.row(2)(0) = kSqrt01_03 * (sh1.row(1)(2) * sh1.row(1)(0) + sh1.row(1)(0) * sh1.row(1)(2)) + -kSqrt01_12 * ((sh1.row(2)(2) * sh1.row(2)(0) + sh1.row(2)(0) * sh1.row(2)(2)) + (sh1.row(0)(2) * sh1.row(0)(0) + sh1.row(0)(0) * sh1.row(0)(2)));
    sh2.row(2)(1) = kSqrt04_03 * sh1.row(1)(1) * sh1.row(1)(0) + -kSqrt01_03 * (sh1.row(2)(1) * sh1.row(2)(0) + sh1.row(0)(1) * sh1.row(0)(0));
    sh2.row(2)(2) = sh1.row(1)(1) * sh1.row(1)(1) + -kSqrt01_04 * (sh1.row(2)(1) * sh1.row(2)(1) + sh1.row(0)(1) * sh1.row(0)(1));
    sh2.row(2)(3) = kSqrt04_03 * sh1.row(1)(1) * sh1.row(1)(2) + -kSqrt01_03 * (sh1.row(2)(1) * sh1.row(2)(2) + sh1.row(0)(1) * sh1.row(0)(2));
    sh2.row(2)(4) = kSqrt01_03 * (sh1.row(1)(2) * sh1.row(1)(2) - sh1.row(1)(0) * sh1.row(1)(0)) + -kSqrt01_12 * ((sh1.row(2)(2) * sh1.row(2)(2) - sh1.row(2)(0) * sh1.row(2)(0)) + (sh1.row(0)(2) * sh1.row(0)(2) - sh1.row(0)(0) * sh1.row(0)(0)));

    sh2.row(3)(0) = kSqrt01_04 * ((sh1.row(1)(2) * sh1.row(2)(0) + sh1.row(1)(0) * sh1.row(2)(2)) + (sh1.row(2)(2) * sh1.row(1)(0) + sh1.row(2)(0) * sh1.row(1)(2)));
    sh2.row(3)(1) = sh1.row(1)(1) * sh1.row(2)(0) + sh1.row(2)(1) * sh1.row(1)(0);
    sh2.row(3)(2) = kSqrt03_04 * (sh1.row(1)(1) * sh1.row(2)(1) + sh1.row(2)(1) * sh1.row(1)(1));
    sh2.row(3)(3) = sh1.row(1)(1) * sh1.row(2)(2) + sh1.row(2)(1) * sh1.row(1)(2);
    sh2.row(3)(4) = kSqrt01_04 * ((sh1.row(1)(2) * sh1.row(2)(2) - sh1.row(1)(0) * sh1.row(2)(0)) + (sh1.row(2)(2) * sh1.row(1)(2) - sh1.row(2)(0) * sh1.row(1)(0)));

    sh2.row(4)(0) = kSqrt01_04 * ((sh1.row(2)(2) * sh1.row(2)(0) + sh1.row(2)(0) * sh1.row(2)(2)) - (sh1.row(0)(2) * sh1.row(0)(0) + sh1.row(0)(0) * sh1.row(0)(2)));
    sh2.row(4)(1) = (sh1.row(2)(1) * sh1.row(2)(0) - sh1.row(0)(1) * sh1.row(0)(0));
    sh2.row(4)(2) = kSqrt03_04 * (sh1.row(2)(1) * sh1.row(2)(1) - sh1.row(0)(1) * sh1.row(0)(1));
    sh2.row(4)(3) = (sh1.row(2)(1) * sh1.row(2)(2) - sh1.row(0)(1) * sh1.row(0)(2));
    sh2.row(4)(4) = kSqrt01_04 * ((sh1.row(2)(2) * sh1.row(2)(2) - sh1.row(2)(0) * sh1.row(2)(0)) - (sh1.row(0)(2) * sh1.row(0)(2) - sh1.row(0)(0) * sh1.row(0)(0)));

    sh3.row(0)(0) = kSqrt01_04 * ((sh1.row(2)(2) * sh2.row(0)(0) + sh1.row(2)(0) * sh2.row(0)(4)) + (sh1.row(0)(2) * sh2.row(4)(0) + sh1.row(0)(0) * sh2.row(4)(4)));
    sh3.row(0)(1) = kSqrt03_02 * (sh1.row(2)(1) * sh2.row(0)(0) + sh1.row(0)(1) * sh2.row(4)(0));
    sh3.row(0)(2) = kSqrt15_16 * (sh1.row(2)(1) * sh2.row(0)(1) + sh1.row(0)(1) * sh2.row(4)(1));
    sh3.row(0)(3) = kSqrt05_06 * (sh1.row(2)(1) * sh2.row(0)(2) + sh1.row(0)(1) * sh2.row(4)(2));
    sh3.row(0)(4) = kSqrt15_16 * (sh1.row(2)(1) * sh2.row(0)(3) + sh1.row(0)(1) * sh2.row(4)(3));
    sh3.row(0)(5) = kSqrt03_02 * (sh1.row(2)(1) * sh2.row(0)(4) + sh1.row(0)(1) * sh2.row(4)(4));
    sh3.row(0)(6) = kSqrt01_04 * ((sh1.row(2)(2) * sh2.row(0)(4) - sh1.row(2)(0) * sh2.row(0)(0)) + (sh1.row(0)(2) * sh2.row(4)(4) - sh1.row(0)(0) * sh2.row(4)(0)));

    sh3.row(1)(0) = kSqrt01_06 * (sh1.row(1)(2) * sh2.row(0)(0) + sh1.row(1)(0) * sh2.row(0)(4)) + kSqrt01_06 * ((sh1.row(2)(2) * sh2.row(1)(0) + sh1.row(2)(0) * sh2.row(1)(4)) + (sh1.row(0)(2) * sh2.row(3)(0) + sh1.row(0)(0) * sh2.row(3)(4)));
    sh3.row(1)(1) = sh1.row(1)(1) * sh2.row(0)(0) + (sh1.row(2)(1) * sh2.row(1)(0) + sh1.row(0)(1) * sh2.row(3)(0));
    sh3.row(1)(2) = kSqrt05_08 * sh1.row(1)(1) * sh2.row(0)(1) + kSqrt05_08 * (sh1.row(2)(1) * sh2.row(1)(1) + sh1.row(0)(1) * sh2.row(3)(1));
    sh3.row(1)(3) = kSqrt05_09 * sh1.row(1)(1) * sh2.row(0)(2) + kSqrt05_09 * (sh1.row(2)(1) * sh2.row(1)(2) + sh1.row(0)(1) * sh2.row(3)(2));
    sh3.row(1)(4) = kSqrt05_08 * sh1.row(1)(1) * sh2.row(0)(3) + kSqrt05_08 * (sh1.row(2)(1) * sh2.row(1)(3) + sh1.row(0)(1) * sh2.row(3)(3));
    sh3.row(1)(5) = sh1.row(1)(1) * sh2.row(0)(4) + (sh1.row(2)(1) * sh2.row(1)(4) + sh1.row(0)(1) * sh2.row(3)(4));
    sh3.row(1)(6) = kSqrt01_06 * (sh1.row(1)(2) * sh2.row(0)(4) - sh1.row(1)(0) * sh2.row(0)(0)) + kSqrt01_06 * ((sh1.row(2)(2) * sh2.row(1)(4) - sh1.row(2)(0) * sh2.row(1)(0)) + (sh1.row(0)(2) * sh2.row(3)(4) - sh1.row(0)(0) * sh2.row(3)(0)));

    sh3.row(2)(0) = kSqrt04_15 * (sh1.row(1)(2) * sh2.row(1)(0) + sh1.row(1)(0) * sh2.row(1)(4)) + kSqrt01_05 * (sh1.row(0)(2) * sh2.row(2)(0) + sh1.row(0)(0) * sh2.row(2)(4)) + -kSqrt1_60 * ((sh1.row(2)(2) * sh2.row(0)(0) + sh1.row(2)(0) * sh2.row(0)(4)) - (sh1.row(0)(2) * sh2.row(4)(0) + sh1.row(0)(0) * sh2.row(4)(4)));
    sh3.row(2)(1) = kSqrt08_05 * sh1.row(1)(1) * sh2.row(1)(0) + kSqrt06_05 * sh1.row(0)(1) * sh2.row(2)(0) + -kSqrt01_10 * (sh1.row(2)(1) * sh2.row(0)(0) - sh1.row(0)(1) * sh2.row(4)(0));
    sh3.row(2)(2) = sh1.row(1)(1) * sh2.row(1)(1) + kSqrt03_04 * sh1.row(0)(1) * sh2.row(2)(1) + -kSqrt01_16 * (sh1.row(2)(1) * sh2.row(0)(1) - sh1.row(0)(1) * sh2.row(4)(1));
    sh3.row(2)(3) = kSqrt08_09 * sh1.row(1)(1) * sh2.row(1)(2) + kSqrt02_03 * sh1.row(0)(1) * sh2.row(2)(2) + -kSqrt01_18 * (sh1.row(2)(1) * sh2.row(0)(2) - sh1.row(0)(1) * sh2.row(4)(2));
    sh3.row(2)(4) = sh1.row(1)(1) * sh2.row(1)(3) + kSqrt03_04 * sh1.row(0)(1) * sh2.row(2)(3) + -kSqrt01_16 * (sh1.row(2)(1) * sh2.row(0)(3) - sh1.row(0)(1) * sh2.row(4)(3));
    sh3.row(2)(5) = kSqrt08_05 * sh1.row(1)(1) * sh2.row(1)(4) + kSqrt06_05 * sh1.row(0)(1) * sh2.row(2)(4) + -kSqrt01_10 * (sh1.row(2)(1) * sh2.row(0)(4) - sh1.row(0)(1) * sh2.row(4)(4));
    sh3.row(2)(6) = kSqrt04_15 * (sh1.row(1)(2) * sh2.row(1)(4) - sh1.row(1)(0) * sh2.row(1)(0)) + kSqrt01_05 * (sh1.row(0)(2) * sh2.row(2)(4) - sh1.row(0)(0) * sh2.row(2)(0)) + -kSqrt1_60 * ((sh1.row(2)(2) * sh2.row(0)(4) - sh1.row(2)(0) * sh2.row(0)(0)) - (sh1.row(0)(2) * sh2.row(4)(4) - sh1.row(0)(0) * sh2.row(4)(0)));

    sh3.row(3)(0) = kSqrt03_10 * (sh1.row(1)(2) * sh2.row(2)(0) + sh1.row(1)(0) * sh2.row(2)(4)) + -kSqrt01_10 * ((sh1.row(2)(2) * sh2.row(3)(0) + sh1.row(2)(0) * sh2.row(3)(4)) + (sh1.row(0)(2) * sh2.row(1)(0) + sh1.row(0)(0) * sh2.row(1)(4)));
    sh3.row(3)(1) = kSqrt09_05 * sh1.row(1)(1) * sh2.row(2)(0) + -kSqrt03_05 * (sh1.row(2)(1) * sh2.row(3)(0) + sh1.row(0)(1) * sh2.row(1)(0));
    sh3.row(3)(2) = kSqrt09_08 * sh1.row(1)(1) * sh2.row(2)(1) + -kSqrt03_08 * (sh1.row(2)(1) * sh2.row(3)(1) + sh1.row(0)(1) * sh2.row(1)(1));
    sh3.row(3)(3) = sh1.row(1)(1) * sh2.row(2)(2) + -kSqrt01_03 * (sh1.row(2)(1) * sh2.row(3)(2) + sh1.row(0)(1) * sh2.row(1)(2));
    sh3.row(3)(4) = kSqrt09_08 * sh1.row(1)(1) * sh2.row(2)(3) + -kSqrt03_08 * (sh1.row(2)(1) * sh2.row(3)(3) + sh1.row(0)(1) * sh2.row(1)(3));
    sh3.row(3)(5) = kSqrt09_05 * sh1.row(1)(1) * sh2.row(2)(4) + -kSqrt03_05 * (sh1.row(2)(1) * sh2.row(3)(4) + sh1.row(0)(1) * sh2.row(1)(4));
    sh3.row(3)(6) = kSqrt03_10 * (sh1.row(1)(2) * sh2.row(2)(4) - sh1.row(1)(0) * sh2.row(2)(0)) + -kSqrt01_10 * ((sh1.row(2)(2) * sh2.row(3)(4) - sh1.row(2)(0) * sh2.row(3)(0)) + (sh1.row(0)(2) * sh2.row(1)(4) - sh1.row(0)(0) * sh2.row(1)(0)));

    sh3.row(4)(0) = kSqrt04_15 * (sh1.row(1)(2) * sh2.row(3)(0) + sh1.row(1)(0) * sh2.row(3)(4)) + kSqrt01_05 * (sh1.row(2)(2) * sh2.row(2)(0) + sh1.row(2)(0) * sh2.row(2)(4)) + -kSqrt1_60 * ((sh1.row(2)(2) * sh2.row(4)(0) + sh1.row(2)(0) * sh2.row(4)(4)) + (sh1.row(0)(2) * sh2.row(0)(0) + sh1.row(0)(0) * sh2.row(0)(4)));
    sh3.row(4)(1) = kSqrt08_05 * sh1.row(1)(1) * sh2.row(3)(0) + kSqrt06_05 * sh1.row(2)(1) * sh2.row(2)(0) + -kSqrt01_10 * (sh1.row(2)(1) * sh2.row(4)(0) + sh1.row(0)(1) * sh2.row(0)(0));
    sh3.row(4)(2) = sh1.row(1)(1) * sh2.row(3)(1) + kSqrt03_04 * sh1.row(2)(1) * sh2.row(2)(1) + -kSqrt01_16 * (sh1.row(2)(1) * sh2.row(4)(1) + sh1.row(0)(1) * sh2.row(0)(1));
    sh3.row(4)(3) = kSqrt08_09 * sh1.row(1)(1) * sh2.row(3)(2) + kSqrt02_03 * sh1.row(2)(1) * sh2.row(2)(2) + -kSqrt01_18 * (sh1.row(2)(1) * sh2.row(4)(2) + sh1.row(0)(1) * sh2.row(0)(2));
    sh3.row(4)(4) = sh1.row(1)(1) * sh2.row(3)(3) + kSqrt03_04 * sh1.row(2)(1) * sh2.row(2)(3) + -kSqrt01_16 * (sh1.row(2)(1) * sh2.row(4)(3) + sh1.row(0)(1) * sh2.row(0)(3));
    sh3.row(4)(5) = kSqrt08_05 * sh1.row(1)(1) * sh2.row(3)(4) + kSqrt06_05 * sh1.row(2)(1) * sh2.row(2)(4) + -kSqrt01_10 * (sh1.row(2)(1) * sh2.row(4)(4) + sh1.row(0)(1) * sh2.row(0)(4));
    sh3.row(4)(6) = kSqrt04_15 * (sh1.row(1)(2) * sh2.row(3)(4) - sh1.row(1)(0) * sh2.row(3)(0)) + kSqrt01_05 * (sh1.row(2)(2) * sh2.row(2)(4) - sh1.row(2)(0) * sh2.row(2)(0)) + -kSqrt1_60 * ((sh1.row(2)(2) * sh2.row(4)(4) - sh1.row(2)(0) * sh2.row(4)(0)) + (sh1.row(0)(2) * sh2.row(0)(4) - sh1.row(0)(0) * sh2.row(0)(0)));

    sh3.row(5)(0) = kSqrt01_06 * (sh1.row(1)(2) * sh2.row(4)(0) + sh1.row(1)(0) * sh2.row(4)(4)) + kSqrt01_06 * ((sh1.row(2)(2) * sh2.row(3)(0) + sh1.row(2)(0) * sh2.row(3)(4)) - (sh1.row(0)(2) * sh2.row(1)(0) + sh1.row(0)(0) * sh2.row(1)(4)));
    sh3.row(5)(1) = sh1.row(1)(1) * sh2.row(4)(0) + (sh1.row(2)(1) * sh2.row(3)(0) - sh1.row(0)(1) * sh2.row(1)(0));
    sh3.row(5)(2) = kSqrt05_08 * sh1.row(1)(1) * sh2.row(4)(1) + kSqrt05_08 * (sh1.row(2)(1) * sh2.row(3)(1) - sh1.row(0)(1) * sh2.row(1)(1));
    sh3.row(5)(3) = kSqrt05_09 * sh1.row(1)(1) * sh2.row(4)(2) + kSqrt05_09 * (sh1.row(2)(1) * sh2.row(3)(2) - sh1.row(0)(1) * sh2.row(1)(2));
    sh3.row(5)(4) = kSqrt05_08 * sh1.row(1)(1) * sh2.row(4)(3) + kSqrt05_08 * (sh1.row(2)(1) * sh2.row(3)(3) - sh1.row(0)(1) * sh2.row(1)(3));
    sh3.row(5)(5) = sh1.row(1)(1) * sh2.row(4)(4) + (sh1.row(2)(1) * sh2.row(3)(4) - sh1.row(0)(1) * sh2.row(1)(4));
    sh3.row(5)(6) = kSqrt01_06 * (sh1.row(1)(2) * sh2.row(4)(4) - sh1.row(1)(0) * sh2.row(4)(0)) + kSqrt01_06 * ((sh1.row(2)(2) * sh2.row(3)(4) - sh1.row(2)(0) * sh2.row(3)(0)) - (sh1.row(0)(2) * sh2.row(1)(4) - sh1.row(0)(0) * sh2.row(1)(0)));

    sh3.row(6)(0) = kSqrt01_04 * ((sh1.row(2)(2) * sh2.row(4)(0) + sh1.row(2)(0) * sh2.row(4)(4)) - (sh1.row(0)(2) * sh2.row(0)(0) + sh1.row(0)(0) * sh2.row(0)(4)));
    sh3.row(6)(1) = kSqrt03_02 * (sh1.row(2)(1) * sh2.row(4)(0) - sh1.row(0)(1) * sh2.row(0)(0));
    sh3.row(6)(2) = kSqrt15_16 * (sh1.row(2)(1) * sh2.row(4)(1) - sh1.row(0)(1) * sh2.row(0)(1));
    sh3.row(6)(3) = kSqrt05_06 * (sh1.row(2)(1) * sh2.row(4)(2) - sh1.row(0)(1) * sh2.row(0)(2));
    sh3.row(6)(4) = kSqrt15_16 * (sh1.row(2)(1) * sh2.row(4)(3) - sh1.row(0)(1) * sh2.row(0)(3));
    sh3.row(6)(5) = kSqrt03_02 * (sh1.row(2)(1) * sh2.row(4)(4) - sh1.row(0)(1) * sh2.row(0)(4));
    sh3.row(6)(6) = kSqrt01_04 * ((sh1.row(2)(2) * sh2.row(4)(4) - sh1.row(2)(0) * sh2.row(4)(0)) - (sh1.row(0)(2) * sh2.row(0)(4) - sh1.row(0)(0) * sh2.row(0)(0)));
    return;
}

Pos PointRotateByAxis(Pos point, Pos center, Vector4f axis, float radian){
    float cost = cos(radian);
    float sint = sin(radian);

    float norm = sqrt(axis(0) * axis(0) + axis(1) * axis(1) + axis(2) * axis(2));
    float x = axis(0) / norm;
    float y = axis(1) / norm;
    float z = axis(2) / norm;

    Pos output;
    output(0) = (x*x*(1-cost)+cost) * point(0) + (x*y*(1-cost)-z*sint) * point(1) + (x*z*(1-cost)+y*sint) * point(2);
    output(1) = (y*x*(1-cost)+z*sint) * point(0) + (y*y*(1-cost)+cost) * point(1) + (y*z*(1-cost)-x*sint) * point(2);
    output(2) = (z*x*(1-cost)-y*sint) * point(0) + (z*y*(1-cost)+x*sint) * point(1) + (z*z*(1-cost)+cost) * point(2);

    float a = center(0);
    float b = center(1);
    float c = center(2);

    output(0) += (a*(y*y+z*z)-x*(b*y+c*z))*(1-cost) + (b*z-c*y)*sint;
    output(1) += (b*(x*x+z*z)-y*(a*x+c*z))*(1-cost) + (c*x-a*z)*sint;
    output(2) += (c*(x*x+y*y)-z*(a*x+b*y))*(1-cost) + (a*y-b*x)*sint;

    return output;
}

std::array<int, 3> OrderIndices(Scale s){
    std::array<float, 3> numbers = {s.scale[0], s.scale[1], s.scale[2]};
    std::array<int, 3> indices = {0, 1, 2};

    std::sort(indices.begin(), indices.end(), [&numbers](int i, int j) { return numbers[i] < numbers[j];});

    return indices;
}

void writeVectorToObj(const std::vector<Pos>& vertices, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    for (const auto& vertex : vertices) {
        file << "v " << vertex.x() << " " << vertex.y() << " " << vertex.z() << "\n";
    }

    file.close();
    std::cout << "File written: " << filename << std::endl;
}

