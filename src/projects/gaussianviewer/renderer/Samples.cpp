#include "Samples.hpp"

#define MAX_ITERS 5
    // vector<SamplePoint> sample_points;
    // vector<SamplePoint> backup_sample_points;

    // vector<vector<unsigned int>> neighbourgs;
    // vector<vector<float>> neighbour_pdfs;

    // Vector1d m_x;
    // std::vector<Vertex> vertices;
    // vector<Pos> positions;
    // std::vector<float> opacity_gs;
    // std::vector<SHs<3>> shs_gs;
    // std::vector<SHs<3>> shs_aim;

void SamplePoints::construct(std::vector<Vertex> verts, vector<Pos> pos, std::vector<float> opacity_vector, std::vector<SHs<3>> shs_vector, vector<SHs<3>> sample_feature_shs){
    vertices = verts;
    positions = pos;
    opacity_gs = opacity_vector;
    shs_gs = shs_vector;
    shs_aim = sample_feature_shs;

    fx_n = shs_aim.size()*3;
    jacobi_n = opacity_gs.size();
    jacobi_m = fx_n;
}

void SamplePoints::gs_adjust(){
    initializeX();
    // double energy = optimize();
}

void SamplePoints::initializeX(){
    m_x.resize(opacity_gs.size());
    for (int i = 0; i < opacity_gs.size(); i++){
        m_x[i] = opacity_gs[i];
    }
}

double SamplePoints::optimize(){
    DefineJacobiStructure(m_jacobi, m_jacobiT);
    SpMat JacTJac;
    Vector1d fx(m_jacobi.rows()), h(m_jacobi.cols()), g(m_jacobi.cols()), fx1(m_jacobi.rows());

    JacTJac = m_jacobiT * m_jacobi;
    Eigen::SimplicialCholesky<SpMat> solver;
    solver.analyzePattern(JacTJac.triangularView<Eigen::Lower>());

    for(int iter=0; iter<MAX_ITERS; iter++){
        auto start_jacobi = std::chrono::steady_clock::now();
        FastCalcJacobiMat(m_jacobi, m_jacobiT);
        auto end_jacobi = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedSeconds = end_jacobi - start_jacobi;
        std::cout << "CalcJacobiMat once took " << elapsedSeconds.count() << " seconds." << std::endl;
        
        auto start_multi = std::chrono::steady_clock::now();
        JacTJac = m_jacobiT * m_jacobi;
        auto end_multi = std::chrono::steady_clock::now();
        elapsedSeconds = end_multi - start_multi;
        std::cout << "JacTJac once took " << elapsedSeconds.count() << " seconds." << std::endl;


        auto start_energy = std::chrono::steady_clock::now();
        CalcEnergyFunc(fx);
        auto end_energy = std::chrono::steady_clock::now();
        elapsedSeconds = end_energy - start_energy;
        std::cout << "CalcEnergyFunc once took " << elapsedSeconds.count() << " seconds." << std::endl;

        auto start_solve = std::chrono::steady_clock::now();
        g = m_jacobiT * (-fx);
		
		solver.compute(JacTJac);

		solver.factorize(JacTJac.triangularView<Eigen::Lower>());
		h = solver.solve(g);

		double normv = m_x.norm();
		double old_energy = fx.dot(fx);

        auto end_solve = std::chrono::steady_clock::now();
        elapsedSeconds = end_solve - start_solve;
        std::cout << "Solving once took " << elapsedSeconds.count() << " seconds." << std::endl;

		for (double alpha = 1; alpha > 1e-15; alpha *= 0.5)
		{
			Vector1d x = m_x + h;
			CalcEnergyFunc(fx1);	//f
			double new_energy = fx1.dot(fx1);
			if (new_energy > old_energy)
				h = h * 0.5;
			else
			{
				m_x = x;
				break;
			}
		}
		double normh = h.norm();

        auto end = std::chrono::steady_clock::now();
        elapsedSeconds = end - start_jacobi;
        std::cout << "One loop totally took " << elapsedSeconds.count() << " seconds." << std::endl;

		if(normh < (normv+real(1e-6)) * real(1e-6))
			break;
	}
	return fx.dot(fx);
}

void SamplePoints::DefineJacobiStructure(SpMat& jacobi, SpMat& jacobiT){
	if(jacobi.rows()!=jacobi_m || jacobi.cols()!=jacobi_n)
	{
		jacobi.resize(jacobi_m,jacobi_n);
		jacobiT.resize(jacobi_n,jacobi_m);
	}
}

void SamplePoints::FastCalcJacobiMat(SpMat& jacobi, SpMat& jacobiT){
    std:vector<Eigen::Triplet<double>> trips;
    // 第一维 能量项数：num_samples*3  第二位 优化量数：num_gs*feature_dim
    for (int i = 0; i < sample_points.size(); i++) {
        for (int j = 0; j < neighbourgs[i].size(); j++) {
            int gs_idx = neighbourgs[i][j];
            trips.push_back(Eigen::Triplet<double>(i*3, gs_idx, neighbour_pdfs[i][j]*shs_gs[gs_idx].shs[0]));
            trips.push_back(Eigen::Triplet<double>(i*3 + 1, gs_idx, neighbour_pdfs[i][j]*shs_gs[gs_idx].shs[1]));
            trips.push_back(Eigen::Triplet<double>(i*3 + 2, gs_idx, neighbour_pdfs[i][j]*shs_gs[gs_idx].shs[2]));
        }
    }
    jacobi.resize(jacobi_m,jacobi_n);
	jacobi.setFromTriplets(trips.begin(),trips.end());
	jacobiT=jacobi.transpose();
}

void SamplePoints::CalcEnergyFunc(Vector1d& fx){
    // 维度 能量项数： num_samples*3
    for (int i = 0; i < sample_points.size(); i++) {
        Vec3 new_shs(0.0, 0.0, 0.0);
        for (int j = 0; j < neighbourgs[i].size(); j++) {
            int gs_idx = neighbourgs[i][j];
            new_shs(0) += neighbour_pdfs[i][j] * shs_gs[gs_idx].shs[0] * m_x[i];
            new_shs(1) += neighbour_pdfs[i][j] * shs_gs[gs_idx].shs[1] * m_x[i];
            new_shs(2) += neighbour_pdfs[i][j] * shs_gs[gs_idx].shs[2] * m_x[i];
        }
        fx(i*3) = new_shs(0) - shs_aim[i].shs[0];
        fx(i*3 + 1) = new_shs(1) - shs_aim[i].shs[1];
        fx(i*3 + 2) = new_shs(2) - shs_aim[i].shs[2];
    }
}