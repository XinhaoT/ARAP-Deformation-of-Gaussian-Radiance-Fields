#include "Deform.hpp"

#define MAX_ITERS 30
#define CONTROL_NODE_NUM 20

void Deform::GetControlNums(){
	control_blocks_num = 0;
	control_nodes_num = 0;
	for (int i = 0; i < (*indices_blocks).size(); i++){
		if ((*blocks_type)[i] != -1){
			control_blocks_num += 1;
			control_nodes_num += (*indices_blocks)[i].size();
		}
	}

	std::cout << "control_blocks_num: " << control_blocks_num << std::endl;
	std::cout << "control_nodes_num: " << control_nodes_num << std::endl;
}

bool Deform::BelongsToSameBlock(unsigned int node_idx1, unsigned int node_idx2){
	for (int i = 0; i < (*indices_blocks).size(); i++){
		if ((*blocks_type)[i] == -1){
			continue;
		}
		auto it1 = find((*indices_blocks)[i].begin(), (*indices_blocks)[i].end(), node_idx1);
		auto it2 = find((*indices_blocks)[i].begin(), (*indices_blocks)[i].end(), node_idx2);
		if ((it1 != (*indices_blocks)[i].end()) && (it2 != (*indices_blocks)[i].end())){
			return true;
		}
	}
	return false;
}

void Deform::SelectKeyControls()
{
	auto start = std::chrono::steady_clock::now();
	for (int i = 0; i < (*indices_blocks).size(); i++){
		if ((*blocks_type)[i] == -1){
			aim_blocks_center.push_back(Pos(0.0, 0.0, 0.0));
			set<unsigned int> empty_set;
			center_control_indices.push_back(empty_set);
			continue;
		}
		else{
			set<unsigned int> cur_indices;
			std::vector<Pos> positions;
			std::vector<unsigned int> c_indices;
			for (int node_idx_i : (*indices_blocks)[i])
			{
				unsigned int i = (*nodes)[node_idx_i].Vertex_index;
				positions.push_back((*nodes)[node_idx_i].Position);
				c_indices.push_back(node_idx_i);
			}
			std::vector<int> selected_indicies = farthest_control_points_sampling(positions, CONTROL_NODE_NUM, false);

			Pos center(0.0, 0.0, 0.0);
			for (int i = 0; i < selected_indicies.size(); i++)
			{
				cur_indices.insert(c_indices[i]);
				center += aim_positions[c_indices[i]];
			}

			center_control_indices.push_back(cur_indices);
			aim_blocks_center.push_back(center / selected_indicies.size());
		}
	}

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsedSeconds = end - start;
	if (_show_deform_efficiency)
	{
		std::cout << "Select Key Controls took " << elapsedSeconds.count() << " seconds." << std::endl;
	}
	return;
}

void Deform::real_time_deform()
{
	setIdentityRots();
	double energy = optimize();
}

void Deform::setIdentityRots()
{
	m_x.resize(each_node_dim * (*free_indices).size());
	m_x.setZero();
	for (int i = 0; i < (*free_indices).size(); i++)
	{
		m_x[i * each_node_dim] = 1.0;
		m_x[i * each_node_dim + 4] = 1.0;
		m_x[i * each_node_dim + 8] = 1.0;
	}
}

double Deform::optimize()
{

	DefineJacobiStructure(m_jacobi, m_jacobiT);
	SpMat JacTJac;
	Vector1d fx(m_jacobi.rows()), h(m_jacobi.cols()), g(m_jacobi.cols()), fx1(m_jacobi.rows());

	JacTJac = m_jacobiT * m_jacobi;
	Eigen::SimplicialCholesky<SpMat> solver;
	solver.analyzePattern(JacTJac.triangularView<Eigen::Lower>());

	for (int iter = 0; iter < MAX_ITERS; iter++)
	{
		if (_show_deform_efficiency)
		{
			std::cout << "Iter " << iter << std::endl;
		}
		if (iter == MAX_ITERS - 1)
		{
			std::cout << "exceed number of maximum iterations" << std::endl;
		}

		auto start = std::chrono::steady_clock::now();
		FastCalcJacobiMat(m_jacobi, m_jacobiT);

		JacTJac = m_jacobiT * m_jacobi;
		CalcEnergyFunc(fx, m_x);
		g = m_jacobiT * (-fx);
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsedSeconds = end - start;
		if (_show_deform_efficiency)
		{
			std::cout << "calc jacobi&energy took " << elapsedSeconds.count() << " seconds." << std::endl;
		}

		start = std::chrono::steady_clock::now();
		solver.compute(JacTJac);
		solver.factorize(JacTJac.triangularView<Eigen::Lower>());
		h = solver.solve(g);
		end = std::chrono::steady_clock::now();
		elapsedSeconds = end - start;
		if (_show_deform_efficiency)
		{
			std::cout << "solver took " << elapsedSeconds.count() << " seconds." << std::endl;
		}

		start = std::chrono::steady_clock::now();
		double normv = m_x.norm();
		double old_energy = fx.dot(fx);
		for (double alpha = 1.0; alpha > 1e-15; alpha *= 0.5)
		{
			Vector1d x = m_x + h;
			CalcEnergyFunc(fx1, x); // f
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
		end = std::chrono::steady_clock::now();
		elapsedSeconds = end - start;
		if (_show_deform_efficiency)
		{
			std::cout << "loop took " << elapsedSeconds.count() << " seconds." << std::endl;
		}

		if (normh < (normv + real(1e-6)) * real(1e-6))
			break;
	}
	return fx.dot(fx);
}

void Deform::DefineJacobiStructure(SpMat &jacobi, SpMat &jacobiT)
{
	if (jacobi.rows() != jacobi_m || jacobi.cols() != jacobi_n)
	{
		jacobi.resize(jacobi_m, jacobi_n);
		jacobiT.resize(jacobi_n, jacobi_m);
	}
}

void Deform::FastCalcJacobiMat(SpMat &jacobi, SpMat &jacobiT)
{
	double cur_weight;
std:
	vector<Eigen::Triplet<double>> trips;

	int index = 0;
	cur_weight = w_rotation;
	int nNode = (*free_indices).size();
	for (int i = 0; i < nNode; i++)
	{
		int k = i * each_node_dim;
		trips.push_back(Eigen::Triplet<double>(index, k + 0, m_x(k + 3) * cur_weight));
		trips.push_back(Eigen::Triplet<double>(index, k + 1, m_x(k + 4) * cur_weight));
		trips.push_back(Eigen::Triplet<double>(index, k + 2, m_x(k + 5) * cur_weight));
		trips.push_back(Eigen::Triplet<double>(index, k + 3, m_x(k + 0) * cur_weight));
		trips.push_back(Eigen::Triplet<double>(index, k + 4, m_x(k + 1) * cur_weight));
		trips.push_back(Eigen::Triplet<double>(index, k + 5, m_x(k + 2) * cur_weight));

		trips.push_back(Eigen::Triplet<double>(index + 1, k + 0, m_x(k + 6) * cur_weight));
		trips.push_back(Eigen::Triplet<double>(index + 1, k + 1, m_x(k + 7) * cur_weight));
		trips.push_back(Eigen::Triplet<double>(index + 1, k + 2, m_x(k + 8) * cur_weight));
		trips.push_back(Eigen::Triplet<double>(index + 1, k + 6, m_x(k + 0) * cur_weight));
		trips.push_back(Eigen::Triplet<double>(index + 1, k + 7, m_x(k + 1) * cur_weight));
		trips.push_back(Eigen::Triplet<double>(index + 1, k + 8, m_x(k + 2) * cur_weight));

		trips.push_back(Eigen::Triplet<double>(index + 2, k + 3, m_x(k + 6) * cur_weight));
		trips.push_back(Eigen::Triplet<double>(index + 2, k + 4, m_x(k + 7) * cur_weight));
		trips.push_back(Eigen::Triplet<double>(index + 2, k + 5, m_x(k + 8) * cur_weight));
		trips.push_back(Eigen::Triplet<double>(index + 2, k + 6, m_x(k + 3) * cur_weight));
		trips.push_back(Eigen::Triplet<double>(index + 2, k + 7, m_x(k + 4) * cur_weight));
		trips.push_back(Eigen::Triplet<double>(index + 2, k + 8, m_x(k + 5) * cur_weight));

		for (int j = 0; j < 3; j++)
		{
			trips.push_back(Eigen::Triplet<double>(index + 3 + j, k + j * 3 + 0, 2 * m_x(k + j * 3 + 0) * cur_weight));
			trips.push_back(Eigen::Triplet<double>(index + 3 + j, k + j * 3 + 1, 2 * m_x(k + j * 3 + 1) * cur_weight));
			trips.push_back(Eigen::Triplet<double>(index + 3 + j, k + j * 3 + 2, 2 * m_x(k + j * 3 + 2) * cur_weight));
		}
		index += 6;
	}

	cur_weight = w_regularization;
	int cur_idx = 0;
	for (unsigned int i : (*free_indices))
	{
		Node node_i = (*nodes)[i];
		Pos vertex_i = node_i.Position;
		int k1 = cur_idx * each_node_dim;
		cur_idx += 1;
		for (int t = 0; t < k_nearest; t++)
		{
			bool flag_update_at_k = true;
			int k;
			int node_idx = node_i.Neighbor[t];
			Pos vertex_k = (*nodes)[node_idx].Position;

			auto it = (*free_indices).find((unsigned int)node_idx);
			if (it == (*free_indices).end())
			{
				flag_update_at_k = false;
			}
			else
			{
				k = std::distance((*free_indices).begin(), it);
			}

			double weight_scale = 1.0;
			if (BelongsToSameBlock(i, node_idx)){
				weight_scale = control_weight_factor;
			}

			for (int j = 0; j < 3; j++)
			{
				trips.push_back(Eigen::Triplet<double>(index + j, k1 + j, (vertex_k(0) - vertex_i(0)) * cur_weight * weight_scale));
				trips.push_back(Eigen::Triplet<double>(index + j, k1 + j + 3, (vertex_k(1) - vertex_i(1)) * cur_weight * weight_scale));
				trips.push_back(Eigen::Triplet<double>(index + j, k1 + j + 6, (vertex_k(2) - vertex_i(2)) * cur_weight * weight_scale));
				trips.push_back(Eigen::Triplet<double>(index + j, k1 + j + 9, cur_weight));

				if (flag_update_at_k)
				{
					trips.push_back(Eigen::Triplet<double>(index + j, k * each_node_dim + j + 9, -cur_weight * weight_scale));
				}
			}
			index += 3;
		}
	}

	// Jacobian terms for static-free nodes pairs, from the static side
	for (unsigned int i : (*static_indices))
	{
		Node node_i = (*nodes)[i];
		for (int t = 0; t < k_nearest; t++)
		{
			bool flag_update_at_k = true;
			int k;
			int node_idx = node_i.Neighbor[t];

			auto it = (*free_indices).find((unsigned int)node_idx);
			if (it == (*free_indices).end())
			{
				flag_update_at_k = false;
			}
			else
			{
				k = std::distance((*free_indices).begin(), it);
			}

			if (flag_update_at_k)
			{
				for (int j = 0; j < 3; j++)
				{
					trips.push_back(Eigen::Triplet<double>(index + j, k * each_node_dim + j + 9, -cur_weight));
				}
			}
			index += 3;
		}
	}

	if (_constraints_on_center)
	{
		cur_weight = w_constraints;
		for (int block_id = 0; block_id < (*indices_blocks).size(); block_id++){
			if ((*blocks_type)[block_id] == -1){continue;}

			for (unsigned int node_idx_i : (center_control_indices)[block_id]){
				unsigned int i = (*nodes)[node_idx_i].Vertex_index;
				Pos vertex_i = (*nodes)[node_idx_i].Position;
				Vertex v_i = (*vertices)[i];
				for (int j = 0; j < k_nearest; j++)
				{
					int node_idx = v_i.Neighbor_Nodes[j];
					double wei = v_i.Neighbor_Weights[j];
					auto it = (*free_indices).find((unsigned int)node_idx);
					if (it == (*free_indices).end())
					{
						continue;
					}

					Pos vertex_k = (*nodes)[node_idx].Position;

					int nid = std::distance((*free_indices).begin(), it); 
					int k1 = nid * each_node_dim;
					for (int k = 0; k < 3; k++)
					{
						trips.push_back(Eigen::Triplet<double>(index + k, k1 + k, wei * (vertex_i(0) - vertex_k(0)) * cur_weight));
						trips.push_back(Eigen::Triplet<double>(index + k, k1 + 3 + k, wei * (vertex_i(1) - vertex_k(1)) * cur_weight));
						trips.push_back(Eigen::Triplet<double>(index + k, k1 + 6 + k, wei * (vertex_i(2) - vertex_k(2)) * cur_weight));
						trips.push_back(Eigen::Triplet<double>(index + k, k1 + 9 + k, wei * cur_weight));
					}
				}
			}
			index += 3;
		}
	}
	else
	{
		cur_weight = w_constraints;
		for (int block_id = 0; block_id < (*indices_blocks).size(); block_id++){
			if ((*blocks_type)[block_id] == -1){continue;}
			
			for (unsigned int node_idx_i : (*indices_blocks)[block_id])
			{
				unsigned int i = (*nodes)[node_idx_i].Vertex_index;
				Pos vertex_i = (*nodes)[node_idx_i].Position;
				Vertex v_i = (*vertices)[i];
				for (int j = 0; j < k_nearest; j++)
				{
					int node_idx = v_i.Neighbor_Nodes[j];
					double wei = v_i.Neighbor_Weights[j];
					auto it = (*free_indices).find((unsigned int)node_idx);
					if (it == (*free_indices).end())
					{
						continue;
					}

					Pos vertex_k = (*nodes)[node_idx].Position;

					int nid = std::distance((*free_indices).begin(), it); // maybe time consuming
					int k1 = nid * each_node_dim;
					for (int k = 0; k < 3; k++)
					{
						trips.push_back(Eigen::Triplet<double>(index + k, k1 + k, wei * (vertex_i(0) - vertex_k(0)) * cur_weight));
						trips.push_back(Eigen::Triplet<double>(index + k, k1 + 3 + k, wei * (vertex_i(1) - vertex_k(1)) * cur_weight));
						trips.push_back(Eigen::Triplet<double>(index + k, k1 + 6 + k, wei * (vertex_i(2) - vertex_k(2)) * cur_weight));
						trips.push_back(Eigen::Triplet<double>(index + k, k1 + 9 + k, wei * cur_weight));
					}
				}
				index += 3;
			}
		}
	}

	jacobi.resize(jacobi_m, jacobi_n);
	jacobi.setFromTriplets(trips.begin(), trips.end());
	jacobiT = jacobi.transpose();
}

void Deform::CalcEnergyFunc(Vector1d &fx, Vector1d &x)
{
	double cur_weight;
	int nNode = (*free_indices).size();
	cur_weight = w_rotation;
	int index = 0;
	for (int i = 0; i < nNode; i++)
	{
		int k = i * each_node_dim;
		Mat3 mat;
		mat << x(k + 0), x(k + 3), x(k + 6),
			x(k + 1), x(k + 4), x(k + 7),
			x(k + 2), x(k + 5), x(k + 8);
		Vec3 c[3];

		for (int j = 0; j < 3; j++)
			c[j] = mat.col(j);

		fx(index + 0) = cur_weight * c[0].dot(c[1]);
		fx(index + 1) = cur_weight * c[0].dot(c[2]);
		fx(index + 2) = cur_weight * c[1].dot(c[2]);
		fx(index + 3) = cur_weight * (c[0].dot(c[0]) - 1.0);
		fx(index + 4) = cur_weight * (c[1].dot(c[1]) - 1.0);
		fx(index + 5) = cur_weight * (c[2].dot(c[2]) - 1.0);

		index += 6;
	}

	cur_weight = w_regularization;
	int cur_idx = 0;
	for (unsigned int i : (*free_indices))
	{
		int k = cur_idx * each_node_dim;
		cur_idx += 1;
		Mat3 mat;
		mat << x(k + 0), x(k + 3), x(k + 6),
			x(k + 1), x(k + 4), x(k + 7),
			x(k + 2), x(k + 5), x(k + 8);
		Vec3 tj(x(k + 9), x(k + 10), x(k + 11));
		Node node_i = (*nodes)[i];
		Pos vertex_i = node_i.Position;
		Vec3 gj(vertex_i(0), vertex_i(1), vertex_i(2));

		for (int t = 0; t < k_nearest; t++)
		{
			int j = node_i.Neighbor[t];
			Pos vertex_k = (*nodes)[j].Position;
			Vec3 tk;

			auto it = (*free_indices).find((unsigned int)j);
			if (it == (*free_indices).end())
			{
				tk = Vec3(0.0, 0.0, 0.0);
			}
			else
			{
				int nid = std::distance((*free_indices).begin(), it);
				int k1 = nid * each_node_dim;
				tk = Vec3(x(k1 + 9), x(k1 + 10), x(k1 + 11));
			}

			double weight_scale = 1.0;
			if (BelongsToSameBlock(i, j)){
				weight_scale = control_weight_factor;
			}

			Vec3 gk(vertex_k(0), vertex_k(1), vertex_k(2));
			Vec3 new_n = mat * (gk - gj) + gj + tj - gk - tk;
			fx(index + 0) = cur_weight * new_n(0) * weight_scale;
			fx(index + 1) = cur_weight * new_n(1) * weight_scale;
			fx(index + 2) = cur_weight * new_n(2) * weight_scale;
			index += 3;
		}
	}

	// Adding the energy for static-free nodes pairs, from the static side
	// Ereg = || R(gk -gj) + gj + tj - gk -tk ||
	// R = I, tj = 0
	// Ereg = |tk|

	for (unsigned int i : (*static_indices))
	{
		Node node_i = (*nodes)[i];
		for (int t = 0; t < k_nearest; t++)
		{
			int k = node_i.Neighbor[t];
			Vec3 tk;
			auto it = (*free_indices).find((unsigned int)k);
			if (it == (*free_indices).end())
			{
				tk = Vec3(0.0, 0.0, 0.0);
			}
			else
			{
				int nid = std::distance((*free_indices).begin(), it);
				int k1 = nid * each_node_dim;
				tk = Vec3(x(k1 + 9), x(k1 + 10), x(k1 + 11));
			}
			Vec3 new_n = -tk;
			fx(index + 0) = cur_weight * new_n(0);
			fx(index + 1) = cur_weight * new_n(1);
			fx(index + 2) = cur_weight * new_n(2);
			index += 3;
		}
	}

	if (_constraints_on_center)
	{
		cur_weight = w_constraints;
		for (int block_id = 0; block_id < (*indices_blocks).size(); block_id++){
			if ((*blocks_type)[block_id] == -1){continue;}

			fx(index + 0) = 0.0;
			fx(index + 1) = 0.0;
			fx(index + 2) = 0.0;
			for (unsigned int node_idx_i : (center_control_indices)[block_id])
			{
				unsigned int i = (*nodes)[node_idx_i].Vertex_index;
				Pos vertex_i = (*nodes)[node_idx_i].Position;
				Vertex v_i = (*vertices)[i];
				Vec3 ve(vertex_i(0), vertex_i(1), vertex_i(2));
				Vec3 new_v(0, 0, 0);

				for (int j = 0; j < k_nearest; j++)
				{
					int node_idx = v_i.Neighbor_Nodes[j];
					double wei = v_i.Neighbor_Weights[j];

					Pos vertex_j = (*nodes)[node_idx].Position;

					auto it = (*free_indices).find((unsigned int)node_idx);
					if (it == (*free_indices).end())
					{
						new_v += wei * ve;
					}
					else
					{
						int nid = std::distance((*free_indices).begin(), it); // maybe time consuming
						int k = nid * each_node_dim;
						Mat3 mat;
						mat << x(k + 0), x(k + 3), x(k + 6),
							x(k + 1), x(k + 4), x(k + 7),
							x(k + 2), x(k + 5), x(k + 8);
						Vec3 gj(vertex_j(0), vertex_j(1), vertex_j(2));
						Vec3 tj(x(k + 9), x(k + 10), x(k + 11));
						new_v += (wei * (mat * (ve - gj) + gj + tj));
					}
				}

				fx(index + 0) += cur_weight * (new_v(0) - aim_blocks_center[block_id](0));
				fx(index + 1) += cur_weight * (new_v(1) - aim_blocks_center[block_id](1));
				fx(index + 2) += cur_weight * (new_v(2) - aim_blocks_center[block_id](2));
			}
			index += 3;
		}
	}
	else
	{
		cur_weight = w_constraints;
		for (int block_id = 0; block_id < (*indices_blocks).size(); block_id++){
			if ((*blocks_type)[block_id] == -1){continue;}

			for (unsigned int node_idx_i : (*indices_blocks)[block_id])
			{
				unsigned int i = (*nodes)[node_idx_i].Vertex_index;
				Pos vertex_i = (*nodes)[node_idx_i].Position;
				Vertex v_i = (*vertices)[i];
				Vec3 ve(vertex_i(0), vertex_i(1), vertex_i(2));
				Vec3 new_v(0, 0, 0);

				for (int j = 0; j < k_nearest; j++)
				{
					int node_idx = v_i.Neighbor_Nodes[j];
					double wei = v_i.Neighbor_Weights[j];

					Pos vertex_j = (*nodes)[node_idx].Position;

					auto it = (*free_indices).find((unsigned int)node_idx);
					if (it == (*free_indices).end())
					{
						new_v += wei * ve;
					}
					else
					{
						int nid = std::distance((*free_indices).begin(), it); // maybe time consuming
						int k = nid * each_node_dim;
						Mat3 mat;
						mat << x(k + 0), x(k + 3), x(k + 6),
							x(k + 1), x(k + 4), x(k + 7),
							x(k + 2), x(k + 5), x(k + 8);
						Vec3 gj(vertex_j(0), vertex_j(1), vertex_j(2));
						Vec3 tj(x(k + 9), x(k + 10), x(k + 11));
						new_v += (wei * (mat * (ve - gj) + gj + tj));
					}
				}

				fx(index + 0) = cur_weight * (new_v(0) - aim_positions[node_idx_i](0));
				fx(index + 1) = cur_weight * (new_v(1) - aim_positions[node_idx_i](1));
				fx(index + 2) = cur_weight * (new_v(2) - aim_positions[node_idx_i](2));
				index += 3;
			}
		}
	}
}
