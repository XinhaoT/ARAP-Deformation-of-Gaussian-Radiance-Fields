/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */

#include <projects/gaussianviewer/renderer/GaussianView.hpp>
#include <core/graphics/GUI.hpp>
#include <thread>
#include <boost/asio.hpp>
#include <imgui_internal.h>
#include <random>

using namespace std;

float sigmoid(const float m1)
{
	return 1.0f / (1.0f + exp(-m1));
}

float inverse_sigmoid(const float m1)
{
	return log(m1 / (1.0f - m1));
}

# define CUDA_SAFE_CALL_ALWAYS(A) \
A; \
cudaDeviceSynchronize(); \
if (cudaPeekAtLastError() != cudaSuccess) \
SIBR_ERR << cudaGetErrorString(cudaGetLastError());

#if DEBUG || _DEBUG
# define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
# define CUDA_SAFE_CALL(A) A
#endif

template<int D>
int loadPly_Origin(const char* filename,
	std::vector<Pos>& pos,
	std::vector<SHs<3>>& shs,
	std::vector<float>& opacities,
	std::vector<Scale>& scales,
	std::vector<Rot>& rot,
	sibr::Vector3f& minn,
	sibr::Vector3f& maxx)
{
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		SIBR_ERR << "Unable to find model's PLY file, attempted:\n" << filename << std::endl;

	// "Parse" header (it has to be a specific format anyway)
	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int count;
	ss >> dummy >> dummy >> count;

	// Output number of Gaussians contained
	SIBR_LOG << "Loading " << count << " Gaussian splats" << std::endl;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	// Read all Gaussians at once (AoS)
	std::vector<RichPoint<D>> points(count);
	infile.read((char*)points.data(), count * sizeof(RichPoint<D>));

	// Resize our SoA data
	pos.resize(count);
	shs.resize(count);
	scales.resize(count);
	rot.resize(count);
	opacities.resize(count);

	// Gaussians are done training, they won't move anymore. Arrange
	// them according to 3D Morton order. This means better cache
	// behavior for reading Gaussians that end up in the same tile 
	// (close in 3D --> close in 2D).
	minn = sibr::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	maxx = -minn;
	for (int i = 0; i < count; i++)
	{
		maxx = maxx.cwiseMax(points[i].pos);
		minn = minn.cwiseMin(points[i].pos);
	}
	std::vector<std::pair<uint64_t, int>> mapp(count);
	for (int i = 0; i < count; i++)
	{
		sibr::Vector3f rel = (points[i].pos - minn).array() / (maxx - minn).array();
		sibr::Vector3f scaled = ((float((1 << 21) - 1)) * rel);
		sibr::Vector3i xyz = scaled.cast<int>();

		uint64_t code = 0;
		for (int i = 0; i < 21; i++) {
			code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
			code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
			code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
		}

		mapp[i].first = code;
		mapp[i].second = i;
	}
	auto sorter = [](const std::pair < uint64_t, int>& a, const std::pair < uint64_t, int>& b) {
		return a.first < b.first;
	};
	std::sort(mapp.begin(), mapp.end(), sorter);

	// Move data from AoS to SoA
	int SH_N = (D + 1) * (D + 1);
	for (int k = 0; k < count; k++)
	{
		int i = mapp[k].second;
		pos[k] = points[i].pos;

		// Normalize quaternion
		float length2 = 0;
		for (int j = 0; j < 4; j++)
			length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
		float length = sqrt(length2);
		for (int j = 0; j < 4; j++)
			rot[k].rot[j] = points[i].rot.rot[j] / length;

		// Exponentiate scale
		for(int j = 0; j < 3; j++)
			scales[k].scale[j] = exp(points[i].scale.scale[j]);

		// Activate alpha
		opacities[k] = sigmoid(points[i].opacity);

		shs[k].shs[0] = points[i].shs.shs[0];
		shs[k].shs[1] = points[i].shs.shs[1];
		shs[k].shs[2] = points[i].shs.shs[2];
		for (int j = 1; j < SH_N; j++)
		{
			shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
			shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
			shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
		}
	}
	return count;
}

// Load the Gaussians from the given file.
template<int D>
int loadPly(const char* filename,
	std::vector<Pos>& pos,
	std::vector<SHs<3>>& shs,
	std::vector<float>& opacities,
	std::vector<Scale>& scales,
	std::vector<Rot>& rot,
	sibr::Vector3f& minn,
	sibr::Vector3f& maxx,
	std::vector<int>& idx)
{
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		SIBR_ERR << "Unable to find model's PLY file, attempted:\n" << filename << std::endl;

	// "Parse" header (it has to be a specific format anyway)
	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int count;
	ss >> dummy >> dummy >> count;

	// Output number of Gaussians contained
	SIBR_LOG << "Loading " << count << " Gaussian splats" << std::endl;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	// Read all Gaussians at once (AoS)
	std::vector<RichPoint_Soup<D>> points(count);
	infile.read((char*)points.data(), count * sizeof(RichPoint_Soup<D>));

	// Resize our SoA data
	pos.resize(count);
	shs.resize(count);
	scales.resize(count);
	rot.resize(count);
	opacities.resize(count);
	idx.resize(count);

	// Gaussians are done training, they won't move anymore. Arrange
	// them according to 3D Morton order. This means better cache
	// behavior for reading Gaussians that end up in the same tile 
	// (close in 3D --> close in 2D).
	minn = sibr::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	maxx = -minn;
	for (int i = 0; i < count; i++)
	{
		maxx = maxx.cwiseMax(points[i].pos);
		minn = minn.cwiseMin(points[i].pos);
	}
	std::vector<std::pair<uint64_t, int>> mapp(count);
	for (int i = 0; i < count; i++)
	{
		sibr::Vector3f rel = (points[i].pos - minn).array() / (maxx - minn).array();
		sibr::Vector3f scaled = ((float((1 << 21) - 1)) * rel);
		sibr::Vector3i xyz = scaled.cast<int>();

		uint64_t code = 0;
		for (int i = 0; i < 21; i++) {
			code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
			code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
			code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
		}

		mapp[i].first = code;
		mapp[i].second = i;
	}
	auto sorter = [](const std::pair < uint64_t, int>& a, const std::pair < uint64_t, int>& b) {
		return a.first < b.first;
	};
	std::sort(mapp.begin(), mapp.end(), sorter);

	// Move data from AoS to SoA
	int SH_N = (D + 1) * (D + 1);
	for (int k = 0; k < count; k++)
	{
		int i = mapp[k].second;
		pos[k] = points[i].pos;

		// Normalize quaternion
		float length2 = 0;
		for (int j = 0; j < 4; j++)
			length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
		float length = sqrt(length2);
		for (int j = 0; j < 4; j++)
			rot[k].rot[j] = points[i].rot.rot[j] / length;

		// Exponentiate scale
		for(int j = 0; j < 3; j++)
			scales[k].scale[j] = exp(points[i].scale.scale[j]);

		// Activate alpha
		opacities[k] = sigmoid(points[i].opacity);

		shs[k].shs[0] = points[i].shs.shs[0];
		shs[k].shs[1] = points[i].shs.shs[1];
		shs[k].shs[2] = points[i].shs.shs[2];
		for (int j = 1; j < SH_N; j++)
		{
			shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
			shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
			shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
		}

		idx[k] = int(round(points[i].index));
		
	}
	return count;
}

void savePly(const char* filename,
	const std::vector<Pos>& pos,
	const std::vector<SHs<3>>& shs,
	const std::vector<float>& opacities,
	const std::vector<Scale>& scales,
	const std::vector<Rot>& rot,
	const sibr::Vector3f& minn,
	const sibr::Vector3f& maxx,
	const std::vector<bool> too_small)
{
	// Read all Gaussians at once (AoS)
	int count = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
			pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
			continue;
		if (too_small[i]) continue;
		count++;
	}
	std::vector<RichPoint<3>> points(count);

	// Output number of Gaussians contained
	SIBR_LOG << "Saving " << count << " Gaussian splats" << std::endl;

	std::ofstream outfile(filename, std::ios_base::binary);

	outfile << "ply\nformat binary_little_endian 1.0\nelement vertex " << count << "\n";

	std::string props1[] = { "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"};
	std::string props2[] = { "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" };

	for (auto s : props1)
		outfile << "property float " << s << std::endl;
	for (int i = 0; i < 45; i++)
		outfile << "property float f_rest_" << i << std::endl;
	for (auto s : props2)
		outfile << "property float " << s << std::endl;
	outfile << "end_header" << std::endl;

	count = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
			pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
			continue;
		if (too_small[i]) continue;
		points[count].pos = pos[i];
		points[count].rot = rot[i];
		// Exponentiate scale
		for (int j = 0; j < 3; j++)
			points[count].scale.scale[j] = log(scales[i].scale[j]);
		// Activate alpha
		points[count].opacity = inverse_sigmoid(opacities[i]);
		points[count].shs.shs[0] = shs[i].shs[0];
		points[count].shs.shs[1] = shs[i].shs[1];
		points[count].shs.shs[2] = shs[i].shs[2];
		for (int j = 1; j < 16; j++)
		{
			points[count].shs.shs[(j - 1) + 3] = shs[i].shs[j * 3 + 0];
			points[count].shs.shs[(j - 1) + 18] = shs[i].shs[j * 3 + 1];
			points[count].shs.shs[(j - 1) + 33] = shs[i].shs[j * 3 + 2];
		}
		count++;
	}
	outfile.write((char*)points.data(), sizeof(RichPoint<3>) * points.size());
}

namespace sibr
{
	// A simple copy renderer class. Much like the original, but this one
	// reads from a buffer instead of a texture and blits the result to
	// a render target. 
	class BufferCopyRenderer
	{

	public:

		BufferCopyRenderer()
		{
			_shader.init("CopyShader",
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.vert"),
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.frag"));

			_flip.init(_shader, "flip");
			_width.init(_shader, "width");
			_height.init(_shader, "height");
		}

		void process(uint bufferID, IRenderTarget& dst, int width, int height, bool disableTest = true)
		{
			if (disableTest)
				glDisable(GL_DEPTH_TEST);
			else
				glEnable(GL_DEPTH_TEST);

			_shader.begin();
			_flip.send();
			_width.send();
			_height.send();

			dst.clear();
			dst.bind();

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);

			sibr::RenderUtility::renderScreenQuad();
			//std::cout << "width: " << width << " height: " << width << std::endl;
			dst.unbind();
			_shader.end();
		}

		/** \return option to flip the texture when copying. */
		bool& flip() { return _flip.get(); }
		int& width() { return _width.get(); }
		int& height() { return _height.get(); }

	private:

		GLShader			_shader; 
		GLuniform<bool>		_flip = false; ///< Flip the texture when copying.
		GLuniform<int>		_width = 1000;
		GLuniform<int>		_height = 800;
	};
}

std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S) {
	auto lambda = [ptr, &S](size_t N) {
		if (N > S)
		{
			if (*ptr)
				CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char*>(*ptr);
	};
	return lambda;
}

sibr::GaussianView::GaussianView(const sibr::BasicIBRScene::Ptr & ibrScene, uint render_w, uint render_h, const char* file, bool* messageRead, int sh_degree, bool white_bg, bool useInterop, int device) :
	_scene(ibrScene),
	_dontshow(messageRead),
	_sh_degree(sh_degree),
	_file(file),
	_white_bg(white_bg),
	_useInterop(useInterop),
	sibr::ViewBase(render_w, render_h)
{
	init(ibrScene, render_w, render_h, file, messageRead, sh_degree, white_bg, useInterop, device);
}

void sibr::GaussianView::init(const sibr::BasicIBRScene::Ptr & ibrScene, uint render_w, uint render_h, const char* file, bool* messageRead, int sh_degree, bool white_bg, bool useInterop, int device){
	int num_devices;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceCount(&num_devices));
	_device = device;
	if (device >= num_devices)
	{
		if (num_devices == 0)
			SIBR_ERR << "No CUDA devices detected!";
		else
			SIBR_ERR << "Provided device index exceeds number of available CUDA devices!";
	}
	CUDA_SAFE_CALL_ALWAYS(cudaSetDevice(device));
	cudaDeviceProp prop;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceProperties(&prop, device));
	if (prop.major < 7)
	{
		SIBR_ERR << "Sorry, need at least compute capability 7.0+!";
	}

	_pointbasedrenderer.reset(new PointBasedRenderer());
	_copyRenderer = new BufferCopyRenderer();
	_copyRenderer->flip() = true;
	_copyRenderer->width() = render_w;
	_copyRenderer->height() = render_h;

	std::vector<uint> imgs_ulr;
	const auto & cams = ibrScene->cameras()->inputCameras();
	for(size_t cid = 0; cid < cams.size(); ++cid) {
		if(cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);

	configfile = std::string(file);
	configfile.replace(configfile.length() - 4, 4, "_config.txt");
	std::cout << "configfile: " << configfile << std::endl;

	std::vector<int> configs;
	if (! std::filesystem::exists(configfile)) {
		configs.push_back(grid_num);
		configs.push_back(is_synthetic);
		configs.push_back(has_soup);
		storeVectorToFile(configs, configfile);
	}
	else {
		configs = loadVectorFromFile(configfile);
		grid_num = configs[0];
		is_synthetic = configs[1];
		int conf_2 = configs[2];
		if (conf_2 == 0){
			has_soup = 0;
			high_quality = false;
		}
		else if(conf_2 == 1){
			has_soup = 1;
			high_quality = true;
		} 
		else if(conf_2 == 2){
			has_soup = 0;
			high_quality = true;
		} 
	}

	std::cout << "high_quality: " << high_quality << std::endl;

	// Load the PLY data (AoS) to the GPU (SoA)
	std::vector<Pos> pos;
	std::vector<Rot> rot;
	std::vector<Scale> scale;
	std::vector<float> opacity;
	std::vector<SHs<3>> shs;
	std::vector<int> index_;

	if (has_soup){
		if (sh_degree == 1)
		{
			count = loadPly<1>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax, index_);
		}
		else if (sh_degree == 2)
		{
			count = loadPly<2>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax, index_);
		}
		else if (sh_degree == 3)
		{
			count = loadPly<3>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax, index_);
		}
	}
	else{
		if (sh_degree == 1)
		{
			count = loadPly_Origin<1>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
		}
		else if (sh_degree == 2)
		{
			count = loadPly_Origin<2>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
		}
		else if (sh_degree == 3)
		{
			count = loadPly_Origin<3>(file, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
		}
	}


	_boxmin = _scenemin;
	_boxmax = _scenemax;

	screen_right = Vector4f(0.01, 0.0, 0.0, 0.0);
	screen_up = Vector4f(0.0, -0.01, 0.0, 0.0);
	screen_forward = Vector4f(0.0, 0.0, 0.01, 0.0);

	global_sp_idx = 0;
	selected_gs_idx = -1;

	flag_rect = false;
	flag_show_nodes = false;
	flag_show_centers = true;
	shadersCompiled = false;
	flag_show_gs =true;
	flag_perform_shs_rot = true;
	flag_test_scale = false;
	flag_show_samples = false;
	show_aim = false;
	show_error = false;
	flag_clip_containing = true;
	show_all_samples = false;

	int P = count;
	show_count = count;

	vertices.resize(P);
	pos_vector.resize(P);
	rot_vector.resize(P);
	scale_vector.resize(P);
	opacity_vector.resize(P);
	shs_vector.resize(P);
	pos_backup_vector.resize(P);

	pos_backup_vector.resize(P);
	rot_backup_vector.resize(P);
	scale_backup_vector.resize(P);
	opacity_backup_vector.resize(P);
	shs_backup_vector.resize(P);

	shs_rot_vector.resize(P);
	too_small.resize(P);

	half_length_vector.resize(P);
	sigma_vector.resize(P);

	global_gs_idx.resize(P);
	influenced_sp_idx.resize(P);

	scale_3d_clip.resize(P);
	scale_3d_max.resize(P);

	indicies.resize(P);

	moved_gaussians = vector<int>(P, 0);
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&moved_gaussians_cuda, sizeof(int) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(moved_gaussians_cuda, 0, sizeof(int) * P));

	for (int i = 0; i < P; i++){
		pos_vector[i] = pos[i];
		rot_vector[i] = rot[i];
		scale_vector[i] = scale[i];
		opacity_vector[i] = opacity[i];
		shs_vector[i] = shs[i];
		vertices[i].index = i;

		pos_backup_vector[i] = pos[i];
		rot_backup_vector[i] = rot[i];
		scale_backup_vector[i] = scale[i];
		opacity_backup_vector[i] = opacity[i];
		shs_backup_vector[i] = shs[i];
		global_gs_idx[i] = false;
		influenced_sp_idx[i] = false;
		if (has_soup){
			indicies[i] = index_[i];
		}
	}




	// Allocate and fill the GPU data
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&pos_cuda, sizeof(Pos) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos_vector.data(), sizeof(Pos) * P, cudaMemcpyHostToDevice));

	auto start = std::chrono::steady_clock::now();

	std::cout << "num_gs: " << pos_vector.size() << std::endl;
	std::cout << "Begin sampling..." << std::endl;
	getOverallAABB();
	num_samples_per_dim = NUM_SAMPLES_PER_DIM;

	std::vector<SamplePoint> sps_vec = getGridSamples();
	static_samples = std::vector<int>(sps_vec.size(), 0);
	static_gaussians = std::vector<bool>(P, false);


	std::cout << "size of sps_vec: " << sps_vec.size() << std::endl;

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos_vector.data(), sizeof(Pos) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rot_cuda, sizeof(Rot) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, rot_vector.data(), sizeof(Rot) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&shs_cuda, sizeof(SHs<3>) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, shs_vector.data(), sizeof(SHs<3>) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opacity_cuda, sizeof(float) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, opacity_vector.data(), sizeof(float) * P, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&scale_cuda, sizeof(Scale) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda, scale_vector.data(), sizeof(Scale) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&shs_highlight_cuda, sizeof(SHs<3>) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_highlight_cuda, shs_vector.data(), sizeof(SHs<3>) * P, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&max_scale_cuda, sizeof(Scale) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(max_scale_cuda, max_scale_vector.data(), sizeof(Scale) * P, cudaMemcpyHostToDevice));

	block_num = 0;
	show_single = false;
	single_gaussian_index = 15;

	pcd_filepath = std::string(file);

	if (is_synthetic){
		std::string meshfile(file);
		meshfile.replace(meshfile.length() - 3, 3, "obj");
		MeshGT->load(meshfile);
		Meshbackup->load(meshfile);

		std::string texfile(file);
		texfile.replace(texfile.length() - 3, 3, "jpg");
		
		if (!imageData.load(texfile)){
			texfile.replace(texfile.length() - 3, 3, "png");
			imageData.load(texfile);
		}

		glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &textureId); 
		glBindTexture(GL_TEXTURE_2D, textureId); 
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageData.w(), imageData.h(), 0, GL_RGB, GL_UNSIGNED_BYTE, imageData.data()); 


		LoadMeshPoints(meshfile, mesh_points);
		LoadMeshPoints(meshfile, mesh_points_backup);
		mesh_vertices.resize(mesh_points.size());
		std::cout << "Size of mesh_points: " << mesh_points.size() << std::endl;
		for (unsigned int i = 0; i < mesh_points.size(); i++){
			mesh_vertices[i].index = i;
		}

		if (has_soup){
			std::string soupfile(file);
			soupfile.replace(soupfile.length() - 4, 4, "_soup.obj");
			LoadMeshPoints(soupfile, soup_points);
			LoadMeshPoints(soupfile, soup_points_backup);
			soup_vertices.resize(soup_points.size());
			for (unsigned int i = 0; i < soup_points.size(); i++){
				soup_vertices[i].index = i;
			}
		}
	}



	std::string txtfile(file);
	txtfile.replace(txtfile.length() - 3, 3, "txt");
	std::cout << "txtfile: " << txtfile << std::endl;

	std::vector<int> node_index;
	std::vector<int> fake(count);

	if (nodes_on_mesh){
		cand_points = simplified_points;
		node_num = simplified_points.size();
	}
	else{
		cand_points = pos_vector;
	}
	cand_points_backup = cand_points;

	node_index = farthest_control_points_sampling(cand_points, node_num, _surface_graph);

	std::vector<Node> nodes;
    for (int i : node_index)
    {
        Node node;
        node.Position = cand_points[i];
        node.Vertex_index = i;
        nodes.push_back(node);
    }
	deform_graph = DeformGraph(nodes, knn_k);

	// // Calculate the adaptive low-pass-filter coefficients
	ada_lpf_vector.resize(grid_count);
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&ada_lpf_ratio, sizeof(Eigen::Matrix3f)*grid_count));
	GetAdaLpfRatio();

	// Calculate the endpoints
	GetEndPoints();


	setupWeights(deform_graph);
	setupWeightsforMesh(deform_graph);
	setupWeightsforSoup(deform_graph);
	setupWeightsforEnds(deform_graph);

	ReloadAimPositions();

	if (nodes_on_mesh){
		cand_vertices = simplified_vertices;
	}
	else {
		cand_vertices = vertices;
	}
	cand_points_backup = cand_points;
	

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&q_vector_cuda, sizeof(Eigen::Quaternionf) * nodes.size()));



	sps = SamplePoints(sps_vec);
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&sample_neighbor_weights, sizeof(float) * sps.sample_points.size() * knn_k));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&sample_neighbor_nodes, sizeof(int) * sps.sample_points.size() * knn_k));
	setupWeightsforSamples(deform_graph);

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&static_samples_cuda, sizeof(int) * sps.sample_points.size()));

	sample_feature_shs.resize(sps.sample_points.size());
	aim_feature_shs.resize(sps.sample_points.size());
	sps.neighbour_pdfs.resize(sps.sample_points.size());
	feature_opacity_vector.resize(sps.sample_points.size());

	auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedSeconds = end - start;
	std::cout << "End sampling..." << std::endl;
    std::cout << "Operation took " << elapsedSeconds.count() << " seconds." << std::endl;



	readObjFile("sphere.obj", v_sphere, n_sphere, f_sphere);
	for (int i = 0; i < v_sphere.size(); i++){ 
		float norm = sqrt(v_sphere[i].x * v_sphere[i].x + v_sphere[i].y * v_sphere[i].y + v_sphere[i].z * v_sphere[i].z);
		v_sphere[i].x = v_sphere[i].x / norm;
		v_sphere[i].y = v_sphere[i].y / norm;
		v_sphere[i].z = v_sphere[i].z / norm;
 	}

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&aim_index_cuda, sizeof(int) * sps.sample_points.size()));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&sample_pos_cuda, sizeof(Pos) * sps.sample_points.size()));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cur_feature_cuda, sizeof(SHs<3>) * sps.sample_points.size()));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&feature_grad_cuda, sizeof(SHs<3>) * sps.sample_points.size()));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opacity_grad_cuda, sizeof(SHs<3>) * sps.sample_points.size()));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&aim_feature_cuda, sizeof(SHs<3>) * sps.sample_points.size()));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cur_opacity_cuda, sizeof(float) * sps.sample_points.size()));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&aim_opacity_cuda, sizeof(float) * sps.sample_points.size()));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&total_feature_loss, sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&total_shape_loss, sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&lr_pos, sizeof(float)));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&dF_dshs_cuda, sizeof(SHs<3>) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&m_shs_cuda, sizeof(SHs<3>) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&v_shs_cuda, sizeof(SHs<3>) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&dF_dopacity_cuda, sizeof(float) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&m_opacity_cuda, sizeof(float) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&v_opacity_cuda, sizeof(float) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&dF_dpos_cuda, sizeof(Pos) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&m_pos_cuda, sizeof(Pos) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&v_pos_cuda, sizeof(Pos) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&dF_drot_cuda, sizeof(Rot) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&m_rot_cuda, sizeof(Rot) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&v_rot_cuda, sizeof(Rot) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&dF_dscale_cuda, sizeof(Scale) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&m_scale_cuda, sizeof(Scale) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&v_scale_cuda, sizeof(Scale) * count));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&dF_dcov3D, sizeof(float) * 9 * count));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opacity_orig_cuda, sizeof(float) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&scale_orig_cuda, sizeof(Scale) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&step, sizeof(int)));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&half_length_cuda, sizeof(float) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&sigma_cuda, sizeof(Eigen::Matrix3f) * P));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&sigma_damp_cuda, sizeof(Eigen::Matrix3f) * P));

	// Create space for view parameters
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&view_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&proj_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cam_pos_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * P * sizeof(int)));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&opt_options_cuda, 12 * sizeof(bool)));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&learning_rate_cuda, 5 * sizeof(float)));

	// Record the gaussians on the surface
	is_surface_vector.resize(count);
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&surface_cuda, P * sizeof(int)));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(surface_cuda, 0, P * sizeof(int)));

	gs_type_vector.resize(count);
	gs_type_vector.assign(count, 0);
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&gs_type_cuda, P * sizeof(int)));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(gs_type_cuda, gs_type_vector.data(), sizeof(int) * count, cudaMemcpyHostToDevice));

	float bg[3] = { white_bg ? 1.f : 0.8f, white_bg ? 1.f : 0.8f, white_bg ? 1.f : 0.8f };
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&empty_grid_cuda, sizeof(int) * valid_grid_num));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(empty_grid_cuda, 0, sizeof(int) * empty_grid.size()));




	gData = new GaussianData(P, 
		(float*)pos.data(),
		(float*)rot.data(),
		(float*)scale.data(),
		opacity.data(),
		(float*)shs.data());

	_gaussianRenderer = new GaussianSurfaceRenderer();

	// Create GL buffer ready for CUDA/GL interop
	glCreateBuffers(1, &imageBuffer);
	glNamedBufferStorage(imageBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

	if (useInterop)
	{
		if (cudaPeekAtLastError() != cudaSuccess)
		{
			SIBR_ERR << "A CUDA error occurred in setup:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
		}
		cudaGraphicsGLRegisterBuffer(&imageBufferCuda, imageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
		useInterop &= (cudaGetLastError() == cudaSuccess);
	}
	if (!useInterop)
	{
		fallback_bytes.resize(render_w * render_h * 3 * sizeof(float));
		cudaMalloc(&fallbackBufferCuda, fallback_bytes.size());
		_interop_failed = true;
	}

	geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
	binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
	imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);

	GPUSetupSamplesFeatures();

	RefreshAdam();
}



void	sibr::GaussianView::ResetAll(){
	for (int i = 0; i < count; i++){
		pos_vector[i] = pos_backup_vector[i];
		rot_vector[i] = rot_backup_vector[i];
		scale_vector[i] = scale_backup_vector[i];
		opacity_vector[i] = opacity_backup_vector[i];
		shs_vector[i] = shs_backup_vector[i];
	}

	GetEndPoints();

	for (int i = 0; i < mesh_points.size(); i++){
		mesh_points[i] = mesh_points_backup[i];
	}

	for (int i = 0; i < simplified_points.size(); i++){
		simplified_points[i] = simplified_points_backup[i];
	}

	for (int i = 0; i < soup_points.size(); i++){
		soup_points[i] = soup_points_backup[i];
	}

	static_indices.clear();
	control_indices.clear();
	indices_blocks.clear();
	aim_centers.clear();
	aim_centers_radius.clear();
	block_num = 0;
	blocks_type.clear();

	deform_graph.static_indices.clear();
	deform_graph.control_indices.clear();
	deform_graph.indices_blocks.clear();
	deform_graph.block_centers.clear();
	deform_graph.history_rot.clear();
	deform_graph.history_trans.clear();
	deform_graph.history_node_positions.clear();
	deform_graph.current_control_idx = 0;

	if (nodes_on_mesh){
		cand_points = simplified_points;
	}
	else{
		cand_points = pos_vector;
	}
	cand_points_backup = cand_points;

	for (unsigned int i = 0; i < deform_graph.nodes.size(); i++){
        deform_graph.free_indices.insert(i);
		deform_graph.nodes[i].Position = cand_points[deform_graph.nodes[i].Vertex_index];
    }

	ReloadAimPositions();

	for (unsigned int i = 0; i < backup_sample_positions.size(); i++){
		sample_positions[i] = backup_sample_positions[i];
    }

	local_gs_idx.clear();

	moved_gaussians = vector<int>(count, 0);

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos_vector.data(), sizeof(Pos) * count, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, rot_vector.data(), sizeof(Rot) * count, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, shs_vector.data(), sizeof(SHs<3>) * count, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, opacity_vector.data(), sizeof(float) * count, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda, scale_vector.data(), sizeof(Scale) * count, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMemset(grid_is_converged_cuda, 0, sizeof(bool) * valid_grid_num));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(grid_nearly_converged_cuda, 0, sizeof(bool) * valid_grid_num));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(empty_grid_cuda, 0, sizeof(int) * empty_grid.size()));
	GetAdaLpfRatio();
	GPUSetupSamplesFeatures();
	RefreshAdam();

	drag_history.clear();
	CheckStaticSamples();
	CheckMovedGaussians();
}

void	sibr::GaussianView::ResetControls(){
	static_indices.clear();
	control_indices.clear();
	indices_blocks.clear();
	aim_centers.clear();
	aim_centers_radius.clear();
	blocks_type.clear();
	block_num = 0;

	deform_graph.static_indices.clear();
	deform_graph.control_indices.clear();
	deform_graph.indices_blocks.clear();
	deform_graph.block_centers.clear();
	deform_graph.current_control_idx = 0;

	for (unsigned int i = 0; i < deform_graph.nodes.size(); i++){
        deform_graph.free_indices.insert(i);
    }
	drag_history.clear();
	CheckStaticSamples();
}

void sibr::GaussianView::setScene(const sibr::BasicIBRScene::Ptr & newScene)
{
	_scene = newScene;

	// Tell the scene we are a priori using all active cameras.
	std::vector<uint> imgs_ulr;
	const auto & cams = newScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid) {
		if (cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
}

void sibr::GaussianView::onRenderIBR(sibr::IRenderTarget & dst, const sibr::Camera & eye)
{
	// std::cout << "update PVM matrix" << std::endl;
	if (currMode == "Ellipsoids")
	{
		_gaussianRenderer->process(count, *gData, eye, dst, 0.2f);
	}
	else if (currMode == "Initial Points")
	{
		_pointbasedrenderer->process(_scene->proxies()->proxy(), eye, dst);
	}
	else
	{
		// To Ortho
		if (_to_ortho){
			eye.setOrthoCam(ortho_scale, ortho_scale);
		}
		// Convert view and projection to target coordinate system
		auto view_mat = eye.view();
		auto proj_mat = eye.viewproj();
		view_mat.row(1) *= -1;
		view_mat.row(2) *= -1;
		proj_mat.row(1) *= -1;

		if (_log_camera){
			std::cout << "view_mat:\n" << view_mat << std::endl;
			std::cout << "proj_mat:\n" << proj_mat << std::endl;
			_log_camera = false;
		}


		T = eye.viewproj() * eye.view() * eye.model();

		// Compute additional view parameters
		float tan_fovy = tan(eye.fovy() * 0.5f);
		float tan_fovx = tan_fovy * eye.aspect();

		// Copy frame-dependent data to GPU
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(view_cuda, view_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(proj_cuda, proj_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(cam_pos_cuda, &eye.position(), sizeof(float) * 3, cudaMemcpyHostToDevice));

		float* image_cuda = nullptr;
		if (!_interop_failed)
		{
			// Map OpenGL buffer resource for use with CUDA
			size_t bytes;
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &imageBufferCuda));
			CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&image_cuda, &bytes, imageBufferCuda));
		}
		else
		{
			image_cuda = fallbackBufferCuda;
		}



		// Rasterize
		int* rects = _fastCulling ? rect_cuda : nullptr;
		float* boxmin = _cropping ? (float*)&_boxmin : nullptr;
		float* boxmax = _cropping ? (float*)&_boxmax : nullptr;

		if ((has_background) && (show_background)){
			CudaRasterizer::Rasterizer::forward(
				geomBufferFunc,
				binningBufferFunc,
				imgBufferFunc,
				show_count, _sh_degree, 16,
				background_cuda,
				_resolution.x(), _resolution.y(),
				all_pos_cuda,
				all_shs_cuda,
				nullptr,
				all_opacity_cuda,
				all_scale_cuda,
				_scalingModifier,
				all_rot_cuda,
				nullptr,
				view_cuda,
				proj_cuda,
				cam_pos_cuda,
				tan_fovx,
				tan_fovy,
				false,
				image_cuda,
				surface_cuda,
				nullptr,
				rects,
				boxmin,
				boxmax,
				_to_ortho,
				ortho_scale
			);
		}
		else {
			CudaRasterizer::Rasterizer::forward(
				geomBufferFunc,
				binningBufferFunc,
				imgBufferFunc,
				show_count, _sh_degree, 16,
				background_cuda,
				_resolution.x(), _resolution.y(),
				pos_cuda,
				shs_highlight_cuda,
				nullptr,
				opacity_cuda,
				scale_cuda,
				_scalingModifier,
				rot_cuda,
				nullptr,
				view_cuda,
				proj_cuda,
				cam_pos_cuda,
				tan_fovx,
				tan_fovy,
				false,
				image_cuda,
				surface_cuda,
				nullptr,
				rects,
				boxmin,
				boxmax,
				_to_ortho,
				ortho_scale
			);
		}
		
		
		if (!_interop_failed)
		{
			// Unmap OpenGL resource for use with OpenGL
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &imageBufferCuda));
		}
		else
		{
			CUDA_SAFE_CALL(cudaMemcpy(fallback_bytes.data(), fallbackBufferCuda, fallback_bytes.size(), cudaMemcpyDeviceToHost));
			glNamedBufferSubData(imageBuffer, 0, fallback_bytes.size(), fallback_bytes.data());
		}
		// Copy image contents to framebuffer
		_copyRenderer->process(imageBuffer, dst, _resolution.x(), _resolution.y());
	}

	if (cudaPeekAtLastError() != cudaSuccess)
	{
		SIBR_ERR << "A CUDA error occurred during rendering:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
	}

	//RenderHelpers(dst);
	_finish_rendering = true;
}

void sibr::GaussianView::onUpdate(Input & input, const Viewport & vp)
{

	if (input.mouseScroll() > 0){
		ortho_scale *= 1.05f;
	}
	if (input.mouseScroll() < 0){
		ortho_scale *= 0.95f;
	}

	Vector2i cur_pos = input.mousePosition();
	// Lets just go here!
	//Vector2f window_size = input.getWindowsize();
	window_size = input._current_window_size;
	if (input.key().isActivated(sibr::Key::M)){
		//std::cout << "M is activated" << std::endl;
		if (input.mouseButton().isPressed(sibr::Mouse::Right)){
			std::cout << "Right is pressed" << std::endl;
			max_x = cur_pos(0);
			max_y = cur_pos(1);
			press_x = cur_pos(0);
			press_y = cur_pos(1);
			flag_rect = true;
		}

		if (input.mouseButton().isReleased(sibr::Mouse::Right)){
			std::cout << "Right is released" << std::endl;
			flag_select = true;
			min_x = min(cur_pos(0), max_x);
			min_y = min(cur_pos(1), max_y);
			max_x = max(cur_pos(0), max_x);
			max_y = max(cur_pos(1), max_y);
			flag_rect = false;

			vector<unsigned int> node_idx_block;
			vector<unsigned int> gs_idx_block;

			Pos center(0.0, 0.0, 0.0);
			for (unsigned int i = 0; i < deform_graph.nodes.size(); i++)
			{
				Vector2f pos_2d = get_2d_pos(deform_graph.nodes[i].Position, T, window_size(0), window_size(1));

				if ((pos_2d(0) > min_x) && (pos_2d(0) < max_x) && (pos_2d(1) > min_y) && (pos_2d(1) < max_y))
				{
					bool not_selected = true;
					for (int block_idx = 0; block_idx < deform_graph.indices_blocks.size(); block_idx++){
						auto it = std::find(deform_graph.indices_blocks[block_idx].begin(), deform_graph.indices_blocks[block_idx].end(), i);
						if (it != deform_graph.indices_blocks[block_idx].end()){
							not_selected = false;
							break;
						}
					}
					if (not_selected){
						node_idx_block.push_back(i);
						gs_idx_block.push_back(deform_graph.nodes[i].Vertex_index);
						center += deform_graph.nodes[i].Position;
					}
				}
			}

			for (int i = 0 ; i < node_idx_block.size() ; i++){
				std::cout << node_idx_block[i] << " ";
			}
			std::cout << std::endl;

			if (gs_idx_block.size()){
				if (block_num == 0) {
					deform_graph.current_control_idx = 0;
					blocks_type.push_back(1);
				}
				else{
					blocks_type.push_back(0);
				}

				block_num += 1;

				indices_blocks.push_back(gs_idx_block);
				deform_graph.indices_blocks.push_back(node_idx_block);
				center = center / gs_idx_block.size();
				deform_graph.block_centers.push_back(center);
				aim_centers.push_back(center);

				std::cout << "center: " << center << std::endl;

				float max_distance = 0.0f;
				for (unsigned int i = 0; i < gs_idx_block.size(); i++) {
					float distance = pts_distance(cand_points[gs_idx_block[i]], center);
					max_distance = max(max_distance, distance);
				}
				aim_centers_radius.push_back(max_distance + 0.03f);

				history.total_operations += 1;
				history.operation_types.push_back(0);
				history.block_nodes.push_back(node_idx_block);

				UpdateIndicies(deform_graph);
			}
			getCentersMesh();
		}
	}


	if (input.key().isActivated(sibr::Key::K)){
		if (input.mouseButton().isPressed(sibr::Mouse::Right)){
			std::cout << "Right is pressed" << std::endl;
			max_x = cur_pos(0);
			max_y = cur_pos(1);
			press_x = cur_pos(0);
			press_y = cur_pos(1);
			flag_rect = true;
		}

		if (input.mouseButton().isReleased(sibr::Mouse::Right)){
			std::cout << "Right is released" << std::endl;
			flag_select = true;
			min_x = min(cur_pos(0), max_x);
			min_y = min(cur_pos(1), max_y);
			max_x = max(cur_pos(0), max_x);
			max_y = max(cur_pos(1), max_y);
			flag_rect = false;


			UpdateCenterRadius();

			for (unsigned int i = 0; i < aim_centers.size(); i++)
			{
				Vector2f pos_2d = get_2d_pos(aim_centers[i], T, window_size(0), window_size(1));

				if ((pos_2d(0) > min_x) && (pos_2d(0) < max_x) && (pos_2d(1) > min_y) && (pos_2d(1) < max_y))
				{
					if (blocks_type[i] == 0){
						blocks_type[i] = 1;
					}
				}
			}
			UpdateIndicies(deform_graph);
		}
	}


	if (input.key().isPressed(sibr::Key::U)){
		for (int i = 0; i < blocks_type.size(); i++) {
			if (blocks_type[i] == 1){
				blocks_type[i] = 0;
			}
		}
		UpdateIndicies(deform_graph);
	}

	if (input.key().isPressed(sibr::Key::C)){
		if (block_num != 0){
			block_num -= 1;

			int cur_idx = deform_graph.current_control_idx;

			indices_blocks.erase(indices_blocks.begin() + cur_idx);
			deform_graph.indices_blocks.erase(deform_graph.indices_blocks.begin() + cur_idx);
			deform_graph.block_centers.erase(deform_graph.block_centers.begin() + cur_idx);
			aim_centers.erase(aim_centers.begin() + cur_idx);
			aim_centers_radius.erase(aim_centers_radius.begin() + cur_idx);
			blocks_type.erase(blocks_type.begin() + cur_idx);

			if (block_num != 0){
				deform_graph.current_control_idx = deform_graph.current_control_idx % block_num;
			}
			else{
				ResetControls();
			}

			UpdateIndicies(deform_graph);
			getCentersMesh();

			// Should ADD the history Part
			history.total_operations += 1;
			history.operation_types.push_back(-(cur_idx + 1));
		}

	}

	if (input.key().isPressed(sibr::Key::N)){
		if (block_num > 0){
			deform_graph.current_control_idx += 1;
			deform_graph.current_control_idx = deform_graph.current_control_idx % block_num;
			for (int i = 0; i < block_num; i++){
				if (blocks_type[i] < 0) continue;
				blocks_type[i] = 0;
			}
			if (blocks_type[deform_graph.current_control_idx] != -1){
				blocks_type[deform_graph.current_control_idx] = 1;
			}
			UpdateIndicies(deform_graph);
		}
		// getCentersMesh();
	}

	if (input.key().isPressed(sibr::Key::J)){
		if (block_num > 0){
			blocks_type[deform_graph.current_control_idx] = -1;
			UpdateIndicies(deform_graph);
		}
		// getCentersMesh();
	}

	if (input.key().isPressed(sibr::Key::V)){
		deform_mode = (deform_mode + 1) % 3;
		switch (deform_mode)
		{
		case 0:
			_is_twist = false;
			_is_scale = false;
			_constraints_on_center = true;
			break;
		case 1:
			_is_twist = true;
			_is_scale = false;
			_constraints_on_center = false;
			break;
		case 2:
			_is_twist = false;
			_is_scale = true;
			_constraints_on_center = false;
			break;
		default:
			break;
		}
	}

	if (input.key().isPressed(sibr::Key::X)){
		_constraints_on_center = !_constraints_on_center;
	}




	if (input.key().isActivated(sibr::Key::B)){

		if (input.mouseButton().isActivated(sibr::Mouse::Left)){
			if (deform_graph.indices_blocks.size()){
				_during_deform = true;
				flag_valid_deform = true;
				auto allstart = std::chrono::steady_clock::now();
				xoffset = cur_pos(0) - lastX;
				yoffset = cur_pos(1) - lastY;

				xoffset = max(-_max_displacement_per_step, min(_max_displacement_per_step, xoffset));
				yoffset = max(-_max_displacement_per_step, min(_max_displacement_per_step, yoffset));

				Vector4f movement_right = T.inverse() * screen_right;
				Vector4f movement_up = T.inverse() * screen_up;
				Vector4f movement_forward = T.inverse() * screen_forward;

				if (!(_is_twist || _is_scale)) {
					Pos movement(0.0, 0.0, 0.0);
					movement += Pos(movement_right(0) * xoffset, movement_right(1) * xoffset, movement_right(2) * xoffset);
					movement += Pos(movement_up(0) * yoffset, movement_up(1) * yoffset, movement_up(2) * yoffset);

					Pos temp_center = aim_centers[deform_graph.current_control_idx];
					Vector4f clip_coords = T*Vector4f(temp_center(0), temp_center(1), temp_center(2), 1.0f);
					float z_ndc = clip_coords(2) / clip_coords(3);

					float x_ndc = (2.0f*(float)cur_pos(0))/(float)window_size(0) -1.0f;
					float y_ndc = 1.0f - (2.0f*(float)cur_pos(1)) / (float)window_size(1);
					clip_coords = Vector4f(x_ndc, y_ndc, z_ndc, 1.0f);
					Vector4f w_coords_now = T.inverse() * clip_coords;

					float x_ndc_last = (2.0f*(float)lastX)/(float)window_size(0) -1.0f;
					float y_ndc_last = 1.0f - (2.0f*(float)lastY) / (float)window_size(1);
					clip_coords = Vector4f(x_ndc_last, y_ndc_last, z_ndc, 1.0f);
					Vector4f w_coords_last = T.inverse() * clip_coords;

					w_coords_now = w_coords_now/w_coords_now(3) - w_coords_last/w_coords_last(3);

					movement = Pos(w_coords_now(0), w_coords_now(1), w_coords_now(2));

					if (fix_x){
						movement(0) = 0.0f;
					}
					if (fix_y){
						movement(1) = 0.0f;
					}
					if (fix_z){
						movement(2) = 0.0f;
					}
					UpdateAimPosition(movement, _record_history);

					if (history.mouse_movements.size() == history.move_operations){
						history.mouse_movements.push_back(vector<Pos>());
						history.twist_axis.push_back(movement_right);
					}
					history.mouse_movements[history.move_operations].push_back(movement);

				}
				else if(_is_scale) {
					UpdateAimPositionScale(yoffset);

					if (history.mouse_movements.size() == history.move_operations){
						history.mouse_movements.push_back(vector<Pos>());
						history.twist_axis.push_back(movement_right);
					}
					history.mouse_movements[history.move_operations].push_back(Pos(yoffset, 0, 0));
				}
				else{
					Vector4f axis;
					Pos v0(0.0, 0.0, 0.0);
					v0 += Pos(movement_right(0) * xoffset, movement_right(1) * xoffset, movement_right(2) * xoffset);
					v0 += Pos(movement_up(0) * yoffset, movement_up(1) * yoffset, movement_up(2) * yoffset);

					Pos v1(movement_forward(0), movement_forward(1), movement_forward(2));

					Pos _axis = (v0.cross(v1)).transpose();
					axis = Vector4f(_axis(0), _axis(1), _axis(2), 0.0);

					int offset = (int)(sqrt(pow(xoffset, 2)+pow(yoffset, 2)));

					if (_axis.norm() > 1e-5){
						UpdateAimPositionTwist(axis, offset, _record_history);

						if (history.mouse_movements.size() == history.move_operations){
							history.mouse_movements.push_back(vector<Pos>());
							history.twist_axis.push_back(axis);
						}
						history.mouse_movements[history.move_operations].push_back(Pos(offset, 0, 0));
					}
				}

				auto start = std::chrono::steady_clock::now();
				Deform deform(cand_vertices, cand_points, control_indices, static_indices, deform_graph, aim_positions, _constraints_on_center, _show_deform_efficiency, knn_k,  blocks_type, _weighting_factor);

				if (_optimize_from_start){
					deform.nodes = &(deform_graph.back_up_nodes);
				}

				deform.real_time_deform();				

				double *x = deform.m_x.data();

				deform_graph.putFreeInputs(x, cand_points);

				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> elapsedSeconds = end - start;
				if (_show_deform_efficiency){
					std::cout << "\nSolve Deformation took " << elapsedSeconds.count() << " seconds." << std::endl;
				}

				auto update_start = std::chrono::steady_clock::now();
				// The order of the update is important!
				if (! _only_deform_gs){
					UpdatePositionforSamples(deform_graph); // before update the nodes' position
				}

				UpdatePosition(deform_graph);
				UpdateAsSixPointsWithdrawBad(deform_graph);
				if (nodes_on_mesh){
					cand_points = simplified_points;
				}
				else{
					cand_points = pos_vector;
				}
				ReloadAimPositions();


				auto update_end = std::chrono::steady_clock::now();
				std::chrono::duration<double> update_elapsedSeconds = update_end - update_start;
				if (_show_deform_efficiency){
					std::cout << "Update Gaussians&Samples took " << update_elapsedSeconds.count() << " seconds." << std::endl;
				}

				deform_graph.resetRT();

				auto allend = std::chrono::steady_clock::now();
				std::chrono::duration<double> allelapsedSeconds = allend - allstart;
				if (_show_deform_efficiency){
					std::cout << "Whole Deformation step took " << allelapsedSeconds.count() << " seconds. \n" << std::endl;
				}
			}
			UpdateCenterRadius();
			getCentersMesh();
		}
	}

	if (playback_steps > 0){

		float direction = 1.0f;
		int step_ = drag_history.size() - playback_steps;

		if (!_drag_direction) {
			direction = -1.0f; 
			step_ = playback_steps - 1;
		}

		UpdateAimPosition(drag_history[step_]*direction, false);

		Deform deform(cand_vertices, cand_points, control_indices, static_indices, deform_graph, aim_positions, _constraints_on_center, _show_deform_efficiency, knn_k, blocks_type, _weighting_factor);

		if (_optimize_from_start){
			deform.nodes = &(deform_graph.back_up_nodes);
		}

		deform.real_time_deform();			
		double *x = deform.m_x.data();
		deform_graph.putFreeInputs(x, cand_points);
		

		if (! _only_deform_gs){
			UpdatePositionforSamples(deform_graph); // before update the nodes' position
		}
		UpdatePosition(deform_graph);
		UpdateAsSixPointsWithdrawBad(deform_graph);
		if (nodes_on_mesh){
			cand_points = simplified_points;
		}
		else{
			cand_points = pos_vector;
		}
		ReloadAimPositions();


		deform_graph.resetRT();

		playback_steps -= 1;
	}


	if (input.key().isReleased(sibr::Key::B) && flag_valid_deform){
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos_vector.data(), sizeof(Pos)*count, cudaMemcpyHostToDevice)); //?

		if (!_only_deform_gs){
			UpdateFeatures();
			RefreshAdam();
		}

		if (_is_twist){
			history.operation_types.push_back(2);
		}
		else if (_is_scale){
			history.operation_types.push_back(3);
		}
		else {
			if (_constraints_on_center){
				history.operation_types.push_back(1);
			}
			else{
				history.operation_types.push_back(4);
			}
		}

		history.blocks_types_moves.push_back(blocks_type);
		if (_constraints_on_center){
			history.energy_on_centers.push_back(1);
		}
		else{
			history.energy_on_centers.push_back(0);
		}
		
		history.total_operations += 1;
		history.move_operations += 1;
		flag_valid_deform = false;
		UpdateCenterRadius();
		getCentersMesh();

		// here check the moved gaussians
		CheckMovedGaussians();
	}

	lastX = cur_pos(0);
	lastY = cur_pos(1);


	if (flag_show_gs){
		if ((has_background) && (show_background)){
			show_count = count + bkg_count;
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(all_pos_cuda, pos_vector.data(), sizeof(Pos) * count, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(all_rot_cuda, rot_vector.data(), sizeof(Rot) * count, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(all_opacity_cuda, opacity_vector.data(), sizeof(float) * count, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(all_scale_cuda, scale_vector.data(), sizeof(Scale) * count, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(all_shs_cuda, shs_vector.data(), sizeof(SHs<3>) * count, cudaMemcpyHostToDevice));
			
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(gs_type_cuda, gs_type_vector.data(), sizeof(int) * count, cudaMemcpyHostToDevice));
			if (flag_show_centers) {
				HighlightSelectedGsCUDA(count, all_shs_cuda, gs_type_cuda, _during_deform, 48, knn_k);
			}
			_during_deform = false;
		}
		else {
			show_count = count;
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos_vector.data(), sizeof(Pos) * count, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, rot_vector.data(), sizeof(Rot) * count, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, shs_vector.data(), sizeof(SHs<3>) * count, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, opacity_vector.data(), sizeof(float) * count, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_cuda, scale_vector.data(), sizeof(Scale) * count, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_highlight_cuda, shs_vector.data(), sizeof(SHs<3>) * count, cudaMemcpyHostToDevice));

			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(gs_type_cuda, gs_type_vector.data(), sizeof(int) * count, cudaMemcpyHostToDevice));
			if (flag_show_centers) {
				HighlightSelectedGsCUDA(count, shs_highlight_cuda, gs_type_cuda, _during_deform, 48, knn_k);
			}
			_during_deform = false;
		}
	}
	else{
		show_count = count;
		vector<float> transparance(count, 0.0);
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_cuda, transparance.data(), sizeof(float) * count, cudaMemcpyHostToDevice));
	}


	if (input.key().isReleased(sibr::Key::F1)){
		TakeSnapshot();
	}
	if (input.key().isReleased(sibr::Key::F2)){
		LoadSnapshot(_snap_idx);
		std::cout << "LoadSnapshot: " << _snap_idx << std::endl;
	}
	if (input.key().isReleased(sibr::Key::G)){
		flag_show_nodes = !flag_show_nodes;
	}
	if (input.key().isReleased(sibr::Key::F)){
		flag_show_gs = !flag_show_gs;
	}
	if (input.key().isReleased(sibr::Key::H)){
		flag_show_centers = !flag_show_centers;
	}

	if (input.key().isReleased(sibr::Key::R)){
		ResetAll();
	}
	if (input.key().isReleased(sibr::Key::T)){
		ResetControls();
	}

	if (input.key().isReleased(sibr::Key::F4)){
		flag_show_samples = !flag_show_samples;
		if (flag_show_samples){
			getSamplesMesh();
		}
	}

	if (input.key().isReleased(sibr::Key::F7)){
		flag_clip_containing = !flag_clip_containing;
	}
	if (input.key().isReleased(sibr::Key::F6)){

		auto start_op = std::chrono::steady_clock::now();

		adjust_op_range = false;
		if (high_quality){
			ExpandOpRange(); // seems un-necessary for common GS	
		}

		if (enable_adaptive_lpf){
			GetAdaLpfRatio();
		}
		CopyFeatureFromCPU2GPU();

		UpdateContainingRelationshipWithStaticGrids();

		for (int i = 0; i < _optimize_steps; i++){

			if (_is_record && (cur_step % _snapshot_interval == 0)){
				CopyFeatureFromGPU2CPU();
				TakeSnapshot();
			}

			if ((i % _containing_relation_interval == 0) && (i != 0)){
				CopyPartialInfoFromGPU2CPU();
				UpdateContainingRelationship();
			}

			GPUOptimize();
		}
		CopyFeatureFromGPU2CPU();

		adjust_op_range = true;
		CUDA_SAFE_CALL_ALWAYS(cudaMemset(current_static_grids_cuda, 0, sizeof(int) * grid_count));

		auto end_op = std::chrono::steady_clock::now();
    	std::chrono::duration<double> elapsedSeconds_op = end_op - start_op;
		std::cout << "Whole Optimization took " << elapsedSeconds_op.count() << " seconds." << std::endl;
	}
	if (input.key().isReleased(sibr::Key::F8)){
		show_aim = !show_aim;
		getSamplesMesh();
	}
	if (input.key().isReleased(sibr::Key::Left)){
		if (snapshots.size() == 0) return;
		_snap_idx -= 1;
		if (_snap_idx < 0){
			_snap_idx = snapshots.size() - 1;
		}
		LoadSnapshot(_snap_idx);
		getSamplesMesh();
		std::cout << "LoadSnapshot: " << _snap_idx << std::endl;
	}
	if (input.key().isReleased(sibr::Key::Right)){
		if (snapshots.size() == 0) return;
		_snap_idx += 1;
		_snap_idx = _snap_idx % snapshots.size();
		LoadSnapshot(_snap_idx);
		getSamplesMesh();
		std::cout << "LoadSnapshot: " << _snap_idx << std::endl;
	}

	if (run_history){
		if (history.operation_types[history_step] < 0){
			if (block_num != 0){
				block_num -= 1;

				int cur_idx = -(history.operation_types[history_step] + 1);
				deform_graph.current_control_idx = cur_idx;

				indices_blocks.erase(indices_blocks.begin() + cur_idx);
				deform_graph.indices_blocks.erase(deform_graph.indices_blocks.begin() + cur_idx);
				deform_graph.block_centers.erase(deform_graph.block_centers.begin() + cur_idx);
				aim_centers.erase(aim_centers.begin() + cur_idx);
				aim_centers_radius.erase(aim_centers_radius.begin() + cur_idx);
				blocks_type.erase(blocks_type.begin() + cur_idx);

				if (block_num != 0){
					deform_graph.current_control_idx = deform_graph.current_control_idx % block_num;
				}
				else{
					ResetControls();
				}

				UpdateIndicies(deform_graph);
				getCentersMesh();
				history_step += 1;
			}
		}

		else if (history.operation_types[history_step] == 0){
			deform_graph.indices_blocks.push_back(history.block_nodes[add_block_op_idx]);

			vector<unsigned int> cur_block;
			Pos center(0.0, 0.0, 0.0);
			for (unsigned int j = 0; j < history.block_nodes[add_block_op_idx].size(); j++){
				cur_block.push_back(deform_graph.nodes[history.block_nodes[add_block_op_idx][j]].Vertex_index);
				center += deform_graph.nodes[history.block_nodes[add_block_op_idx][j]].Position;
			}
			center = center / cur_block.size();
			aim_centers.push_back(center);
			deform_graph.block_centers.push_back(center);

			float max_distance = 0.0f;
			for (unsigned int j = 0; j < cur_block.size(); j++) {
				float distance = pts_distance(cand_points[deform_graph.nodes[history.block_nodes[add_block_op_idx][j]].Vertex_index], center);
				max_distance = max(max_distance, distance);
			}
			aim_centers_radius.push_back(max_distance + 0.03f);
			indices_blocks.push_back(cur_block);
			if (block_num == 0) {
				deform_graph.current_control_idx = 0;
				blocks_type.push_back(1);
			}
			else{
				blocks_type.push_back(0);
			}

			block_num += 1;
			UpdateIndicies(deform_graph);

			add_block_op_idx += 1;
			history_step += 1;
		}

		else if ((history.operation_types[history_step] > 0) && (!run_move)){
			blocks_type = history.blocks_types_moves[mouse_move_op_idx];
			UpdateIndicies(deform_graph);

			cur_move_step = 0;
			run_move = true;
		}

		if (run_move && (cur_move_step < history.mouse_movements[mouse_move_op_idx].size())){
			_during_deform = true;
			auto allstart = std::chrono::steady_clock::now();

			auto start = std::chrono::steady_clock::now();
			if (history.operation_types[history_step] == 1){
				UpdateAimPosition(history.mouse_movements[mouse_move_op_idx][cur_move_step], _record_history);
				_constraints_on_center = true;
			}
			else if (history.operation_types[history_step] == 2){
				UpdateAimPositionTwist(history.twist_axis[mouse_move_op_idx], (int)history.mouse_movements[mouse_move_op_idx][cur_move_step](0), _record_history);
				_constraints_on_center = false;
			}
			else if (history.operation_types[history_step] == 3){
				UpdateAimPositionScale((int)history.mouse_movements[mouse_move_op_idx][cur_move_step](0));
				_constraints_on_center = false;
			}
			else if (history.operation_types[history_step] == 4){
				UpdateAimPosition(history.mouse_movements[mouse_move_op_idx][cur_move_step], _record_history);
				_constraints_on_center = false;
			}
			auto end = std::chrono::steady_clock::now();

			std::chrono::duration<double> elapsedSeconds = end - start;
			

			start = std::chrono::steady_clock::now();
			Deform deform(cand_vertices, cand_points, control_indices, static_indices, deform_graph, aim_positions, _constraints_on_center, _show_deform_efficiency, knn_k, blocks_type, _weighting_factor);

			if (_optimize_from_start){
				deform.nodes = &(deform_graph.back_up_nodes);
			}

			deform.real_time_deform();				
			double *x = deform.m_x.data();
			deform_graph.putFreeInputs(x, cand_points);
			end = std::chrono::steady_clock::now();
			elapsedSeconds = end - start;
			if (_show_deform_efficiency){
				std::cout << "\nSolve Deformation took " << elapsedSeconds.count() << " seconds." << std::endl;
			}

			// The order of the update is important!
			start = std::chrono::steady_clock::now();
			if (!_only_deform_gs){
				UpdatePositionforSamples(deform_graph); // before update the nodes' position
			}
			UpdatePosition(deform_graph);
			UpdateAsSixPointsWithdrawBad(deform_graph);
			if (nodes_on_mesh){
				cand_points = simplified_points;
			}
			else{
				cand_points = pos_vector;
			}
			ReloadAimPositions();
			end = std::chrono::steady_clock::now();
			elapsedSeconds = end - start;
			if (_show_deform_efficiency){
				std::cout << "Update Gaussians&Samples took " << elapsedSeconds.count() << " seconds." << std::endl;
			}

			deform_graph.resetRT();

			auto allend = std::chrono::steady_clock::now();
			std::chrono::duration<double> allelapsedSeconds = allend - allstart;
			if (_show_deform_efficiency){
				std::cout << "Whole step took " << allelapsedSeconds.count() << " seconds. \n" << std::endl;
			}

			cur_move_step += 1;
		}
		if (run_move && (cur_move_step == history.mouse_movements[mouse_move_op_idx].size())){
			// here check the moved gaussians
			CheckMovedGaussians();
			run_move = false;
			mouse_move_op_idx += 1;
			history_step += 1;
		}

		if (history_step == history.total_operations){
			run_history = false;
			UpdateFeatures();
		}
		UpdateCenterRadius();
		getCentersMesh();


	}

	if (run_script){
		if (script_step < deform_script.node_aims.size()){
			_during_deform = true;
			auto allstart = std::chrono::steady_clock::now();
			int aim_node_idx = 0;
			// Update aim positions
			for (int block_idx = 0; block_idx < block_num; block_idx++){
				if (blocks_type[block_idx] == 0) {
					for (int i = 0; i < deform_graph.indices_blocks[block_idx].size(); i++){
						unsigned int node_idx = deform_graph.indices_blocks[block_idx][i];
						aim_positions[node_idx] = temp_aim_nodes[node_idx].Position;
					}
				}
				else if (blocks_type[block_idx] == 1){
					for (int i = 0; i < deform_graph.indices_blocks[block_idx].size(); i++){
						aim_positions[deform_graph.indices_blocks[block_idx][i]] = deform_script.node_aims[script_step][aim_node_idx];
						aim_node_idx += 1;
					}
				}
			}

			auto start = std::chrono::steady_clock::now();
			Deform deform(cand_vertices, cand_points, control_indices, static_indices, deform_graph, aim_positions, _constraints_on_center, _show_deform_efficiency, knn_k, blocks_type, _weighting_factor);

			if (_optimize_from_start){
				deform.nodes = &(deform_graph.back_up_nodes);
			}

			deform.real_time_deform();				
			double *x = deform.m_x.data();
			deform_graph.putFreeInputs(x, cand_points);
			auto end = std::chrono::steady_clock::now();
			std::chrono::duration<double> elapsedSeconds = end - start;
			if (_show_deform_efficiency){
				std::cout << "\nSolve Deformation took " << elapsedSeconds.count() << " seconds." << std::endl;
			}

			// The order of the update is important!
			start = std::chrono::steady_clock::now();
			if (!_only_deform_gs){
				UpdatePositionforSamples(deform_graph); // before update the nodes' position
			}
			UpdatePosition(deform_graph);
			UpdateAsSixPointsWithdrawBad(deform_graph);
			if (nodes_on_mesh){
				cand_points = simplified_points;
			}
			else{
				cand_points = pos_vector;
			}
			ReloadAimPositions();
			end = std::chrono::steady_clock::now();
			elapsedSeconds = end - start;
			if (_show_deform_efficiency){
				std::cout << "Update Gaussians&Samples took " << elapsedSeconds.count() << " seconds." << std::endl;
			}

			deform_graph.resetRT();

			auto allend = std::chrono::steady_clock::now();
			std::chrono::duration<double> allelapsedSeconds = allend - allstart;
			if (_show_deform_efficiency){
				std::cout << "Whole step took " << allelapsedSeconds.count() << " seconds. \n" << std::endl;
			}

			script_step += 1;
		} 

		if (script_step == deform_script.node_aims.size()){
			// here check the moved gaussians
			CheckMovedGaussians();
			run_script = false;
			UpdateFeatures();
		}
		UpdateCenterRadius();
	}
}

void sibr::GaussianView::UpdateIndicies(DeformGraph& dm){
	if (block_num == 0) return;

	for (int i = 0; i < dm.nodes.size(); i++){
		dm.free_indices.insert(i);
	}
	dm.static_indices.clear();
	dm.control_indices.clear();

	static_indices.clear();
	control_indices.clear();

	for (int i = 0; i < block_num; i++){
		if (blocks_type[i] < 0){
			for (int j = 0; j < dm.indices_blocks[i].size(); j++){
				dm.free_indices.erase(dm.indices_blocks[i][j]);	
				dm.static_indices.insert(dm.indices_blocks[i][j]);
			}
			for (int j = 0; j < indices_blocks[i].size(); j++){
				static_indices.insert(indices_blocks[i][j]);	
			}
		}
	}

	CheckStaticSamples();
}


void sibr::GaussianView::CheckStaticSamples(){
	std::fill(static_samples.begin(), static_samples.end(), 1);
	#pragma omp parallel for
	for (int i = 0; i < sample_positions.size(); i++) {
		SamplePoint s = sps.sample_points[i];
		for (int j = 0; j < deform_graph.k_nearest; j++){
			unsigned int node_idx = s.Neighbor_Nodes[j];
			if (deform_graph.static_indices.find(node_idx) == deform_graph.static_indices.end()){
				static_samples[i] = 0;
			}
		}
	}

	std::fill(static_gaussians.begin(), static_gaussians.end(), true);
	std::fill(gs_type_vector.begin(), gs_type_vector.end(), 0);

	if (deform_graph.indices_blocks.size() != 0){
		#pragma omp parallel for
		for (int i = 0; i < ends_vertices.size(); i++) {
			//  loop over all the endpoints of gaussians
			int gs_idx = i / 6;
			Vertex end = ends_vertices[i];

			for (int j = 0; j < deform_graph.k_nearest; j++){
				unsigned int node_idx = end.Neighbor_Nodes[j];
				if (deform_graph.static_indices.find(node_idx) == deform_graph.static_indices.end()){
					static_gaussians[gs_idx] = false;
				}
			}
		}
	}
	
	if (deform_graph.indices_blocks.size() != 0){
		#pragma omp parallel for
		for (int i = 0; i < ends_vertices.size(); i++) {
			//  loop over all the endpoints of gaussians
			int gs_idx = i / 6;
			Vertex end = ends_vertices[i];

			for (int j = 0; j < deform_graph.k_nearest; j++){
				unsigned int node_idx = end.Neighbor_Nodes[j];
				for (int k = 0; k < blocks_type.size(); k++) { // loop over all the blocks of nodes
					if (blocks_type[k] < 0){
						if (std::find(deform_graph.indices_blocks[k].begin(), deform_graph.indices_blocks[k].end(), node_idx) != deform_graph.indices_blocks[k].end()){
							gs_type_vector[gs_idx] -= 1;
						}}
					else if (blocks_type[k] == 1){
						if (std::find(deform_graph.indices_blocks[k].begin(), deform_graph.indices_blocks[k].end(), node_idx) != deform_graph.indices_blocks[k].end()){
							gs_type_vector[gs_idx] += 1;
						}
					}
					else if (blocks_type[k] == 0){
						if (std::find(deform_graph.indices_blocks[k].begin(), deform_graph.indices_blocks[k].end(), node_idx) != deform_graph.indices_blocks[k].end()){
							gs_type_vector[gs_idx] += (knn_k * 6 * 2); // (knn_k * 6) the total count of the connections of one Gaussian's endpoints with the nodes
						} 
					}
				}
			}
		}
	}

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(gs_type_cuda, gs_type_vector.data(), sizeof(int) * count, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(static_samples_cuda, static_samples.data(), sizeof(int) * sample_positions.size(), cudaMemcpyHostToDevice));

	return;
}

void sibr::GaussianView::CheckMovedGaussians(){
	if (deform_graph.indices_blocks.size() != 0){
		#pragma omp parallel for
		for (int i = 0; i < ends_vertices.size(); i++) {
			//  loop over all the endpoints of gaussians
			int gs_idx = i / 6;
			Vertex end = ends_vertices[i];

			for (int j = 0; j < deform_graph.k_nearest; j++){
				unsigned int node_idx = end.Neighbor_Nodes[j];
				if (deform_graph.static_indices.find(node_idx) == deform_graph.static_indices.end()){
					moved_gaussians[gs_idx] = 1;
				}
			}
		}
	}
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(moved_gaussians_cuda, moved_gaussians.data(), sizeof(int) * count, cudaMemcpyHostToDevice));
}


void sibr::GaussianView::onGUI()
{
	// Generate and update UI elements
	const std::string guiName = "3D Gaussians";
	if (ImGui::Begin(guiName.c_str())) 
	{
		if (ImGui::BeginCombo("Render Mode", currMode.c_str()))
		{
			if (ImGui::Selectable("Splats"))
				currMode = "Splats";
			if (ImGui::Selectable("Initial Points"))
				currMode = "Initial Points";
			if (ImGui::Selectable("Ellipsoids"))
				currMode = "Ellipsoids";
			ImGui::EndCombo();
		}
	}
	if (currMode == "Splats")
	{
		ImGui::SliderFloat("Scaling Modifier", &_scalingModifier, 0.001f, 1.0f);
	}
	ImGui::Checkbox("Fast culling", &_fastCulling);

	ImGui::Checkbox("Crop Box", &_cropping);
	if (_cropping)
	{
		ImGui::SliderFloat("Box Min X", &_boxmin.x(), _scenemin.x(), _scenemax.x());
		ImGui::SliderFloat("Box Min Y", &_boxmin.y(), _scenemin.y(), _scenemax.y());
		ImGui::SliderFloat("Box Min Z", &_boxmin.z(), _scenemin.z(), _scenemax.z());
		ImGui::SliderFloat("Box Max X", &_boxmax.x(), _scenemin.x(), _scenemax.x());
		ImGui::SliderFloat("Box Max Y", &_boxmax.y(), _scenemin.y(), _scenemax.y());
		ImGui::SliderFloat("Box Max Z", &_boxmax.z(), _scenemin.z(), _scenemax.z());
		ImGui::InputText("File", _buff, 512);
		if (ImGui::Button("Save"))
		{
			std::vector<Pos> pos(count);
			std::vector<Rot> rot(count);
			std::vector<float> opacity(count);
			std::vector<SHs<3>> shs(count);
			std::vector<Scale> scale(count);
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos.data(), pos_cuda, sizeof(Pos) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot.data(), rot_cuda, sizeof(Rot) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity.data(), opacity_cuda, sizeof(float) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs.data(), shs_cuda, sizeof(SHs<3>) * count, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale.data(), scale_cuda, sizeof(Scale) * count, cudaMemcpyDeviceToHost));

			for (int i = 0; i < count; i++){
				too_small[i] = false;
			}
			for (int i = 0; i < count; i++){
				if (scale[i].scale[0] * scale[i].scale[1] * scale[i].scale[2] > 1e-6){
					too_small[i] = false;
				}
			}
			savePly(_buff, pos, shs, opacity, scale, rot, _boxmin, _boxmax, too_small);
		}
	}

	ImGui::End();

	if(!*_dontshow && !accepted && _interop_failed)
		ImGui::OpenPopup("Error Using Interop");

	if (!*_dontshow && !accepted && _interop_failed && ImGui::BeginPopupModal("Error Using Interop", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::SetItemDefaultFocus();
		ImGui::SetWindowFontScale(2.0f);
		ImGui::Text("This application tries to use CUDA/OpenGL interop.\n"\
			" It did NOT work for your current configuration.\n"\
			" For highest performance, OpenGL and CUDA must run on the same\n"\
			" GPU on an OS that supports interop.You can try to pass a\n"\
			" non-zero index via --device on a multi-GPU system, and/or try\n" \
			" attaching the monitors to the main CUDA card.\n"\
			" On a laptop with one integrated and one dedicated GPU, you can try\n"\
			" to set the preferred GPU via your operating system.\n\n"\
			" FALLING BACK TO SLOWER RENDERING WITH CPU ROUNDTRIP\n");

		ImGui::Separator();

		if (ImGui::Button("  OK  ")) {
			ImGui::CloseCurrentPopup();
			accepted = true;
		}
		ImGui::SameLine();
		ImGui::Checkbox("Don't show this message again", _dontshow);
		ImGui::EndPopup();
	}

	ImGui::Begin("Optimizing Options");
	ImGui::Checkbox("Show Loss", &_optimize_options[6]);
	ImGui::Checkbox("Boundary Energy", &_optimize_options[7]);
	ImGui::SameLine();
	ImGui::Checkbox("Opacity Energy", &_optimize_options[8]);
	ImGui::SameLine();
	ImGui::Checkbox("Low-pass Filter", &_optimize_options[9]);
	ImGui::Checkbox("Adaptive Low-pass Filter", &enable_adaptive_lpf);
	ImGui::Checkbox("Show Shape Loss", &_optimize_options[10]);
	ImGui::Checkbox("Only Optimize Deformed Region", &_optimize_options[11]);
	ImGui::InputInt("Containing Relation Interval", &_containing_relation_interval, 1, 100);



	ImGui::InputInt("Optimize Step", &_optimize_steps, 1, 200);
	ImGui::InputInt("Load Snapshot", &_snap_idx, 1, 50);
	ImGui::InputInt("Snapshot Interval", &_snapshot_interval, 1, 100);

	ImGui::Checkbox("Is Recording", &_is_record);

	if (ImGui::Button("Clear SnapShots")) {
		snapshots.clear();
		_snap_idx = 0;
		RefreshAdam();
	}

	ImGui::SliderFloat("ALPHA Cutoff Display", &_show_alpha_cutoff, 0.0f, 0.99f);
	ImGui::Checkbox("Enable Alpha", &_enable_alpha);
	ImGui::End();


	ImGui::Begin("Deform Options");
	ImGui::InputFloat("w_Rot", &(_w_rot), 1.0, 10.0, 4);
	ImGui::InputFloat("w_Reg", &(_w_reg), 1.0, 10.0, 4);
	ImGui::InputFloat("w_Con", &(_w_con), 1.0, 10.0, 4);
	if(ImGui::Button("Set Weights")){
		deform_graph.w_rot = (double) _w_rot;
		deform_graph.w_reg = (double) _w_reg;
		deform_graph.w_con = (double) _w_con;
	}

	ImGui::SameLine();
	ImGui::Checkbox("Only Deform Gs", &_only_deform_gs);

	ImGui::Checkbox("Deform Efficiency", &_show_deform_efficiency);

	ImGui::End();


	ImGui::Begin("Key Infos");
	ImGui::Text("Num Gaussians %d", count);
	ImGui::Text("Num Samples %d", sps.sample_points.size());
	ImGui::Text("Total Nodes %d", deform_graph.nodes.size());
	ImGui::Text("Free Nodes %d", deform_graph.free_indices.size());
	ImGui::Text("Handle Constraints %d", control_indices.size());
	ImGui::Text("KNN %d", knn_k);
	ImGui::Checkbox("Ortho Camera", &_to_ortho);
	if(ImGui::Button("Log Camera")){
		_log_camera = true;
	}
	ImGui::End();

	ImGui::Begin("Reset Deform Graph");
	ImGui::InputInt("Node Maximum", &node_num, 1, 5000);
	if (ImGui::Button("Rebuild Deform Graph")) {
		RebuildGraph();
		drag_history.clear();
	}
	ImGui::SameLine();
	if (ImGui::Checkbox("Build Graph on Mesh", &nodes_on_mesh)){
		if (nodes_on_mesh){
			node_num = simplified_points.size();
		}
		RebuildGraph();
		drag_history.clear();
	}
	if (ImGui::Button("Load Mesh For Graph")) {
		input_graph_path = "";
		if (sibr::showFilePicker(input_graph_path, Default)) {
			if (!input_graph_path.empty()) {
				LoadMeshForGraph();
			}
		}
	}
	if (ImGui::Button("Store Graph")) {
		std::string output_graph(pcd_filepath);
		output_graph.replace(output_graph.length()-4, 4, "_graph.obj");

		vector<Pos> temp_graph_pos;
		for (Node node : deform_graph.nodes)
		{
			int i = node.Vertex_index;
			temp_graph_pos.push_back(cand_points[i]);
		}
		writeVectorToObj(temp_graph_pos, output_graph);
	}
	ImGui::InputInt("new KNN", &new_knn_k, 1, 5000);
	ImGui::SameLine();
	if (ImGui::Button("Set new kNN")) {
		knn_k = new_knn_k;
		deform_graph.k_nearest = knn_k;
		RebuildGraph();
	}

	ImGui::Checkbox("Record History", &_record_history);
	
	ImGui::Checkbox("Is Twisting", &_is_twist);
	ImGui::Checkbox("Is Scaling", &_is_scale);
	ImGui::Checkbox("Constraints on Centers", &_constraints_on_center);
	ImGui::InputInt("Max Disp Per Step", &_max_displacement_per_step, 1, 10);
	ImGui::End();

	ImGui::Begin("Deformation");
	ImGui::InputText("File", _deform_filepath, 512);
	if (ImGui::Button("Record Deformation")) {
		RecordDeformation();
	}
	ImGui::SameLine();
	if (ImGui::Button("Load Deformation")) {
		input_deform_path = "";
		if (sibr::showFilePicker(input_deform_path, Default)) {
			if (!input_deform_path.empty()) {
				LoadDeformation();
			}
		}
	}
	if (ImGui::Button("Load Deformation w/o Rebuild")) {
		input_deform_path = "";
		if (sibr::showFilePicker(input_deform_path, Default)) {
			if (!input_deform_path.empty()) {
				LoadDeformation_wo_rebuild();
			}
		}
	}
	if (ImGui::Button("Run the Input Deformation")){
		ResetAll();
		RunHistoricalDeform();
	}
	if (ImGui::Button("Continue the Input Deformation")){
		RunHistoricalDeform();
	}

	if (ImGui::Button("Load Deform Script0")) {
		LoadDeformScript0();
	}
	ImGui::SameLine();
	if (ImGui::Button("Load Deform Script1")) {
		LoadDeformScript1();
	}
	if (ImGui::Button("Load Deform Script2")) {
		LoadDeformScript2();
	}
	ImGui::SameLine();
	if (ImGui::Button("Load Deform Script3")) {
		LoadDeformScript3();
	}
	if (ImGui::Button("Load Deform Script4")) {
		LoadDeformScript4();
	}
	ImGui::SameLine();
	if (ImGui::Button("Load Deform Script5")) {
		LoadDeformScript5();
	}
	if (ImGui::Button("Load Deform Script6")) {
		LoadDeformScript6();
	}
	if (ImGui::Button("Run Deform Script")){
		RunDeformScript();
	}

	if (ImGui::Button("Clean Deform History")){
		CleanDeformHistory();
	}
	ImGui::End();

	ImGui::Begin("Background Color");
	ImGui::SliderFloat("bg_color_r", &_bg_color[0], 0.0f, 1.0f);
	ImGui::SliderFloat("bg_color_g", &_bg_color[1], 0.0f, 1.0f);
	ImGui::SliderFloat("bg_color_b", &_bg_color[2], 0.0f, 1.0f);
	if (ImGui::Button("Change Background Color")){
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, _bg_color, 3 * sizeof(float), cudaMemcpyHostToDevice));
	}
	ImGui::End();

	ImGui::Begin("Background Gaussians");
	ImGui::Checkbox("show_background", &show_background);

	// if (ImGui::Button("Load Background Gaussians")){ 
	// 	background_path = "";
	// 	if (sibr::showFilePicker(background_path, Default)) {
	// 		if (!background_path.empty()) {
	// 			LoadBackgroundGaussians();
	// 		}
	// 	}
	// }
	// ImGui::End();
}


void sibr::GaussianView::LoadBackgroundGaussians(){
	if (has_background) {
		return;
	}

	bkg_count = loadPly_Origin<3>(background_path.c_str(), bkg_pos, bkg_shs, bkg_opacity, bkg_scale, bkg_rot, _scenemin, _scenemax);

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&all_pos_cuda, sizeof(Pos) * (count + bkg_count)));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(all_pos_cuda + 3*count, bkg_pos.data(), sizeof(Pos) * bkg_count, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&all_rot_cuda, sizeof(Rot) * (count + bkg_count)));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(all_rot_cuda + 4*count, bkg_rot.data(), sizeof(Rot) * bkg_count, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&all_shs_cuda, sizeof(SHs<3>) * (count + bkg_count)));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(all_shs_cuda + 48*count, bkg_shs.data(), sizeof(SHs<3>) * bkg_count, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&all_opacity_cuda, sizeof(float) * (count + bkg_count)));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(all_opacity_cuda + count, bkg_opacity.data(), sizeof(float) * bkg_count, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&all_scale_cuda, sizeof(Scale)*(count + bkg_count)));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(all_scale_cuda + 3*count, bkg_scale.data(), sizeof(Scale) * bkg_count, cudaMemcpyHostToDevice));


	has_background = true;
	return;
}


sibr::GaussianView::~GaussianView()
{
	// Cleanup
	std::cout << "here delete the gaussianview" << std::endl;
	cudaFree(pos_cuda);
	cudaFree(rot_cuda);
	cudaFree(scale_cuda);
	cudaFree(opacity_cuda);
	cudaFree(shs_cuda);

	CUDA_SAFE_CALL_ALWAYS(cudaFree(view_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(proj_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(cam_pos_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(background_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(opt_options_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(learning_rate_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(opacity_orig_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(scale_orig_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(cur_opacity_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(aim_opacity_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(aim_feature_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(cur_feature_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(feature_grad_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(opacity_grad_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(dF_dopacity_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(dF_dshs_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(dF_dpos_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(dF_drot_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(dF_dscale_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(dF_dcov3D));

	CUDA_SAFE_CALL_ALWAYS(cudaFree(m_opacity_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(v_opacity_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(m_shs_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(v_shs_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(m_pos_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(v_pos_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(m_rot_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(v_rot_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(m_scale_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(v_scale_cuda));

	CUDA_SAFE_CALL_ALWAYS(cudaFree(total_feature_loss));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(total_shape_loss));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(rect_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(step));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(aim_index_cuda));

	CUDA_SAFE_CALL_ALWAYS(cudaFree(sample_pos_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(half_length_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(sigma_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(sigma_damp_cuda));

	CUDA_SAFE_CALL_ALWAYS(cudaFree(grid_gs_prefix_sum_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(valid_grid_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(grid_is_converged_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(grid_nearly_converged_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(grid_loss_sums_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(grided_gs_idx_cuda));

	CUDA_SAFE_CALL_ALWAYS(cudaFree(sample_neighbor_weights));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(sample_neighbor_nodes));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(q_vector_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(current_static_grids_cuda));


	if (!_interop_failed)
	{
		cudaGraphicsUnregisterResource(imageBufferCuda);
	}
	else
	{
		cudaFree(fallbackBufferCuda);
	}
	glDeleteBuffers(1, &imageBuffer);

	if (geomPtr)
		cudaFree(geomPtr);
	if (binningPtr)
		cudaFree(binningPtr);
	if (imgPtr)
		cudaFree(imgPtr);

	delete _copyRenderer;
}

void sibr::GaussianView::LoadDeformScript0(){
	int inter_steps = 50;

	deform_script = DeformScript();

	vector<unsigned int> block1 = {0, 20, 53, 59, 63, 64, 67, 68, 145, 167, 189, 190, 192, 196, 197, 199};
	vector<unsigned int> block2 = {1, 7, 21, 26, 54, 70, 80, 127, 176, 178, 179, 181, 183, 184, 185, 186};

	deform_script.block_nodes.push_back(block1);
	deform_script.block_nodes.push_back(block2);

	deform_script.blocks_types = {0, 1};

	Pos center_start(0.0f, -1.5f, 0.0f);
	vector<Pos> start_node_pos;
	for (int i = 0; i < block2.size(); i++){
		start_node_pos.push_back(deform_graph.nodes[block2[i]].Position);
	}
	

	float k = 5.4f;
	float a = 3.0*Pi*Pi/(k*k);

	Vector4f axis(0.0f, 0.0f, 1.0f, 0.0f);

	for(int df_step = 1; df_step < inter_steps+1; df_step++){
		vector<Pos> aim_cur_step;
		for(int i = 0; i < block2.size(); i++){

			float x = (float)df_step*(-k/Pi)/(float)inter_steps;
			float y = a*x*x; // the offset b counts in the rotation part
			Pos current_center(x, y, 0.0f);
			float radian = -(float)df_step*Pi/(float)inter_steps;

			Pos rotate_output = PointRotateByAxis(start_node_pos[i], center_start, axis, radian) + current_center;
			aim_cur_step.push_back(rotate_output);
		}
		deform_script.node_aims.push_back(aim_cur_step);
	}

	return;
}

void sibr::GaussianView::LoadDeformScript1(){
	int inter_steps = 50;

	deform_script = DeformScript();

	vector<unsigned int> block1 = {0, 20, 53, 59, 63, 64, 67, 68, 145, 167, 189, 190, 192, 196, 197, 199};
	vector<unsigned int> block2 = {1, 7, 21, 26, 54, 70, 80, 127, 176, 178, 179, 181, 183, 184, 185, 186};

	deform_script.block_nodes.push_back(block1);
	deform_script.block_nodes.push_back(block2);

	deform_script.blocks_types = {0, 1};

	Pos center_start(0.0f, -1.5f, 0.0f);
	vector<Pos> start_node_pos;
	for (int i = 0; i < block2.size(); i++){
		start_node_pos.push_back(deform_graph.nodes[block2[i]].Position);
	}
	
	Vector4f axis(0.0f, 1.0f, 0.0f, 0.0f);

	for(int df_step = 1; df_step < inter_steps+1; df_step++){
		vector<Pos> aim_cur_step;
		for(int i = 0; i < block2.size(); i++){

			float radian = -(float)df_step*1.5f*Pi/(float)inter_steps;

			Pos rotate_output = PointRotateByAxis(start_node_pos[i], center_start, axis, radian);
			aim_cur_step.push_back(rotate_output);
		}
		deform_script.node_aims.push_back(aim_cur_step);
	}

	return;
}

void sibr::GaussianView::LoadDeformScript2(){
	int inter_steps = 50;

	deform_script = DeformScript();

	vector<unsigned int> block1 = {0, 20, 53, 59, 63, 64, 67, 68, 145, 167, 189, 190, 192, 196, 197, 199};
	vector<unsigned int> block2 = {1, 7, 21, 26, 54, 70, 80, 127, 176, 178, 179, 181, 183, 184, 185, 186};

	deform_script.block_nodes.push_back(block1);
	deform_script.block_nodes.push_back(block2);

	deform_script.blocks_types = {0, 1};

	Pos center_start(0.0f, -1.5f, 0.0f);
	vector<Pos> start_node_pos;
	for (int i = 0; i < block2.size(); i++){
		start_node_pos.push_back(deform_graph.nodes[block2[i]].Position);
	}
	

	float k = 3.0f;
	float a = Pi/k - Pi*Pi/(2*k);

	Vector4f axis(0.0f, 0.0f, 1.0f, 0.0f);

	for(int df_step = 1; df_step < inter_steps+1; df_step++){
		vector<Pos> aim_cur_step;
		for(int i = 0; i < block2.size(); i++){

			float x = (float)df_step*(-k/Pi)/(float)inter_steps;
			float y = -a*x*x; // the offset b counts in the rotation part
			Pos current_center(x, y, 0.0f);
			float radian = -(float)df_step*(Pi/2.0f)/(float)inter_steps;

			Pos rotate_output = PointRotateByAxis(start_node_pos[i], center_start, axis, radian) + current_center;
			aim_cur_step.push_back(rotate_output);
		}
		deform_script.node_aims.push_back(aim_cur_step);
	}

	return;
}

void sibr::GaussianView::LoadDeformScript3(){
	int inter_steps = 50;

	deform_script = DeformScript();

	vector<unsigned int> block1 = {0, 20, 53, 59, 63, 64, 67, 68, 145, 167, 189, 190, 192, 196, 197, 199};
	vector<unsigned int> block2 = {1, 7, 21, 26, 54, 70, 80, 127, 176, 178, 179, 181, 183, 184, 185, 186};

	deform_script.block_nodes.push_back(block1);
	deform_script.block_nodes.push_back(block2);

	deform_script.blocks_types = {1, 0};

	Pos center_start(0.0f, 1.5f, 0.0f);
	vector<Pos> start_node_pos;
	for (int i = 0; i < block1.size(); i++){
		start_node_pos.push_back(deform_graph.nodes[block1[i]].Position);
	}
	

	float k = 3.0f;
	float a = Pi/k - Pi*Pi/(2*k);

	Vector4f axis(0.0f, 0.0f, 1.0f, 0.0f);

	for(int df_step = 1; df_step < inter_steps+1; df_step++){
		vector<Pos> aim_cur_step;
		for(int i = 0; i < block1.size(); i++){

			float x = (float)df_step*(-k/Pi)/(float)inter_steps;
			float y = a*x*x; // the offset b counts in the rotation part
			Pos current_center(x, y, 0.0f);
			float radian = (float)df_step*(Pi/2.0f)/(float)inter_steps;

			Pos rotate_output = PointRotateByAxis(start_node_pos[i], center_start, axis, radian) + current_center;
			aim_cur_step.push_back(rotate_output);
		}
		deform_script.node_aims.push_back(aim_cur_step);
	}

	return;
}

void sibr::GaussianView::LoadDeformScript4(){
	int inter_steps = 50;

	deform_script = DeformScript();

	vector<unsigned int> block1 = {0, 20, 53, 59, 63, 64, 67, 68, 145, 167, 189, 190, 192, 196, 197, 199};
	vector<unsigned int> block2 = {1, 7, 21, 26, 54, 70, 80, 127, 176, 178, 179, 181, 183, 184, 185, 186};

	deform_script.block_nodes.push_back(block1);
	deform_script.block_nodes.push_back(block2);

	deform_script.blocks_types = {0, 1};

	Pos center_start(0.0f, -1.5f, 0.0f);
	vector<Pos> start_node_pos;
	for (int i = 0; i < block2.size(); i++){
		start_node_pos.push_back(deform_graph.nodes[block2[i]].Position);
	}
	

	float k = 8.4f;
	float a = 3.0*Pi*Pi/(k*k);

	Vector4f axis(0.0f, 0.0f, 1.0f, 0.0f);

	for(int df_step = 1; df_step < inter_steps-10; df_step++){
		vector<Pos> aim_cur_step;
		for(int i = 0; i < block2.size(); i++){

			float x = (float)df_step*(-k/Pi)/(float)inter_steps;
			float y = a*x*x; // the offset b counts in the rotation part
			Pos current_center(x, y, 0.0f);
			float radian = -(float)df_step*Pi/(float)inter_steps;

			Pos rotate_output = PointRotateByAxis(start_node_pos[i], center_start, axis, radian) + current_center;
			aim_cur_step.push_back(rotate_output);
		}
		deform_script.node_aims.push_back(aim_cur_step);
	}

	return;
}

void sibr::GaussianView::LoadDeformScript5(){
	int inter_steps = 50;

	deform_script = DeformScript();

	vector<unsigned int> block1 = {0, 20, 53, 59, 63, 64, 67, 68, 145, 167, 189, 190, 192, 196, 197, 199};
	vector<unsigned int> block2 = {1, 7, 21, 26, 54, 70, 80, 127, 176, 178, 179, 181, 183, 184, 185, 186};

	deform_script.block_nodes.push_back(block1);
	deform_script.block_nodes.push_back(block2);

	deform_script.blocks_types = {0, 1};

	Pos center_start(0.0f, -1.5f, 0.0f);
	vector<Pos> start_node_pos;
	for (int i = 0; i < block2.size(); i++){
		start_node_pos.push_back(deform_graph.nodes[block2[i]].Position);
	}
	
	Vector4f axis(0.0f, 1.0f, 0.0f, 0.0f);

	for(int df_step = 1; df_step < inter_steps-10; df_step++){
		vector<Pos> aim_cur_step;
		for(int i = 0; i < block2.size(); i++){

			float radian = -(float)df_step*1.5f*Pi/(float)inter_steps;

			Pos rotate_output = PointRotateByAxis(start_node_pos[i], center_start, axis, radian);
			aim_cur_step.push_back(rotate_output);
		}
		deform_script.node_aims.push_back(aim_cur_step);
	}

	return;
}


void sibr::GaussianView::LoadDeformScript6(){
	int inter_steps = 50;

	deform_script = DeformScript();

	vector<unsigned int> block1 = {0, 3, 4, 7, 9, 11, 13, 14, 16, 25, 30, 31, 35, 36, 43, 44, 45, 47, 56, 57, 58, 64, 79, 81, 84, 85, 86, 95, 98, 101, 105, 128, 129, 130, 131, 132, 147, 149, 151, 152, 153, 156, 157, 159, 160, 161, 162, 164, 165, 166, 169, 170, 172, 173, 174, 175, 183, 184, 186, 187, 195, 196, 198, 199};
	vector<unsigned int> block2 = {1, 2, 5, 6, 8, 10, 12, 15, 17, 22, 23, 24, 28, 32, 38, 39, 48, 53, 54, 55, 61, 65, 67, 72, 73, 75, 82, 87, 90, 91, 92, 102, 106, 107, 108, 109, 111, 112, 117, 118, 120, 121, 123, 124, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 176, 177, 179, 180, 188, 189, 191, 192};

	deform_script.block_nodes.push_back(block1);
	deform_script.block_nodes.push_back(block2);

	deform_script.blocks_types = {0, 1};

	vector<Pos> start_node_pos;
	for (int i = 0; i < block2.size(); i++){
		start_node_pos.push_back(deform_graph.nodes[block2[i]].Position);
	}
	
	Vector4f axis(0.0f, 1.0f, 0.0f, 0.0f);

	for(int df_step = 1; df_step < inter_steps+1; df_step++){
		vector<Pos> aim_cur_step;
		for(int i = 0; i < block2.size(); i++){

			Pos pos_output = start_node_pos[i] - Pos(0.01f, 0.0f, 0.0f)*df_step;
			aim_cur_step.push_back(pos_output);
		}
		deform_script.node_aims.push_back(aim_cur_step);
	}

	return;
}

void sibr::GaussianView::RunDeformScript(){
	block_num = deform_script.block_nodes.size();
	for (int i = 0; i < block_num; i++){
		vector<unsigned int> gs_idx_block;
		Pos center(0.0, 0.0, 0.0); 

		for (unsigned int j : deform_script.block_nodes[i]){
			gs_idx_block.push_back(deform_graph.nodes[j].Vertex_index);
			center += deform_graph.nodes[j].Position;
		}
		indices_blocks.push_back(gs_idx_block);
		deform_graph.block_centers.push_back(center);
		aim_centers.push_back(center);
		aim_centers_radius.push_back(0.03f);
	}

	deform_graph.indices_blocks = deform_script.block_nodes;
	blocks_type = deform_script.blocks_types;

	UpdateIndicies(deform_graph);

	run_script = true;
	temp_aim_nodes = deform_graph.nodes;
	script_step = 0;

	return;
}

void sibr::GaussianView::setupWeights(const DeformGraph& dm){
	#pragma omp parallel for
	for (unsigned int i = 0; i < pos_vector.size(); i++){
		array<unsigned int, KNN_MAX+1> idx;
		array<double, KNN_MAX> weights;
		dm.computeWeights(pos_vector[i], weights, idx);
		for (unsigned int j = 0; j < knn_k; j++){
			vertices[i].Neighbor_Nodes[j] = idx[j];
			vertices[i].Neighbor_Weights[j] = weights[j];
		} 
	}
	if (!nodes_on_mesh){
		cand_vertices = vertices;
	}
}

void sibr::GaussianView::setupWeightsforMesh(const DeformGraph& dm){
	#pragma omp parallel for
	for (unsigned int i = 0; i < mesh_vertices.size(); i++){
		array<unsigned int, KNN_MAX+1> idx;
		array<double, KNN_MAX> weights;
		dm.computeWeights(mesh_points[i], weights, idx);
		for (unsigned int j = 0; j < knn_k; j++){
			mesh_vertices[i].Neighbor_Nodes[j] = idx[j];
			mesh_vertices[i].Neighbor_Weights[j] = weights[j];
		} 
	}

	#pragma omp parallel for
	for (unsigned int i = 0; i < simplified_vertices.size(); i++){
		array<unsigned int, KNN_MAX+1> idx;
		array<double, KNN_MAX> weights;
		dm.computeWeights(simplified_points[i], weights, idx);
		for (unsigned int j = 0; j < knn_k; j++){
			simplified_vertices[i].Neighbor_Nodes[j] = idx[j];
			simplified_vertices[i].Neighbor_Weights[j] = weights[j];
		} 
	}

	if (nodes_on_mesh){
		cand_vertices = simplified_vertices;
	}
}

void sibr::GaussianView::setupWeightsforSoup(const DeformGraph& dm){
	#pragma omp parallel for
	for (unsigned int i = 0; i < soup_vertices.size(); i++){
		array<unsigned int, KNN_MAX+1> idx;
		array<double, KNN_MAX> weights;
		dm.computeWeights(soup_points[i], weights, idx);
		for (unsigned int j = 0; j < knn_k; j++){
			soup_vertices[i].Neighbor_Nodes[j] = idx[j];
			soup_vertices[i].Neighbor_Weights[j] = weights[j];
		} 
	}
}

void sibr::GaussianView::setupWeightsforEnds(const DeformGraph& dm){
	ends_vertices.resize(ends_vector.size()*6);
	#pragma omp parallel for
	for (unsigned int i = 0; i < ends_vertices.size(); i++){
		array<unsigned int, KNN_MAX+1> idx;
		array<double, KNN_MAX> weights;
		int end_idx = i / 6;
		if ((i % 2) == 0){
			dm.computeWeights(ends_vector[end_idx].ends[(i % 6) / 2].first, weights, idx);
		}
		else{
			dm.computeWeights(ends_vector[end_idx].ends[(i % 6) / 2].second, weights, idx);
		}
		for (unsigned int j = 0; j < knn_k; j++){
			ends_vertices[i].Neighbor_Nodes[j] = idx[j];
			ends_vertices[i].Neighbor_Weights[j] = weights[j];
		} 
	}
}

void sibr::GaussianView::setupWeightsforSamples(const DeformGraph& dm){
	#pragma omp parallel for
	for (unsigned int i = 0; i < sample_positions.size(); i++){
		array<unsigned int, KNN_MAX+1> idx;
		array<double, KNN_MAX> weights;
		dm.computeWeights(sample_positions[i], weights, idx);
		for (unsigned int j = 0; j < knn_k; j++){
			sps.sample_points[i].Neighbor_Nodes[j] = idx[j];
			sps.sample_points[i].Neighbor_Weights[j] = weights[j];
		} 
	}

	vector<float> sp_neighbor_weights(knn_k*sample_positions.size());
	vector<int> sp_neighbor_nodes(knn_k*sample_positions.size());
	#pragma omp parallel for
	for (unsigned int i = 0; i < sample_positions.size(); i++) {
		for (unsigned int j = 0; j < knn_k; j++) {
			sp_neighbor_nodes[i*knn_k + j] = sps.sample_points[i].Neighbor_Nodes[j];
			sp_neighbor_weights[i*knn_k + j] = (float)sps.sample_points[i].Neighbor_Weights[j];
		}
	}
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(sample_neighbor_weights, sp_neighbor_weights.data(), sizeof(float) * knn_k*sample_positions.size(), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(sample_neighbor_nodes, sp_neighbor_nodes.data(), sizeof(int) * knn_k*sample_positions.size(), cudaMemcpyHostToDevice));
}

void sibr::GaussianView::UpdateAimPosition(Pos delta_pos, bool record_history) {
	
	if (record_history){
		drag_history.push_back(delta_pos);
	}

	for (int block_idx = 0; block_idx < deform_graph.indices_blocks.size(); block_idx++){
		if (blocks_type[block_idx] == 1){
			for (int i = 0; i < deform_graph.indices_blocks[block_idx].size(); i++){
				aim_positions[deform_graph.indices_blocks[block_idx][i]] += delta_pos;
			}
		}
	}
}

void sibr::GaussianView::UpdateAimPositionTwist(Vector4f axis, int y, bool record_history){
	float radian = 0.005f * (float)y;

	Pos center(0.0, 0.0, 0.0);
	int count = 0;

	for (int block_idx = 0; block_idx < deform_graph.indices_blocks.size(); block_idx++){
		if (blocks_type[block_idx] == 1){
			for (unsigned int i : deform_graph.indices_blocks[block_idx]){
				center += deform_graph.nodes[i].Position;
				count += 1;
			}
		}
	}
	center = center / count;

	for (int block_idx = 0; block_idx < deform_graph.indices_blocks.size(); block_idx++){
		if (blocks_type[block_idx] == 1){
			for (unsigned int i : deform_graph.indices_blocks[block_idx]){
				aim_positions[i] = PointRotateByAxis(aim_positions[i], center, axis, radian);
			}
		}
	}

}

void sibr::GaussianView::UpdateAimPositionScale(int y){
	float s = 0.002f * (float)y + 1.0f;

	Pos center(0.0, 0.0, 0.0);
	int count = 0;
	for (int block_idx = 0; block_idx < deform_graph.indices_blocks.size(); block_idx++){
		if (blocks_type[block_idx] == 1){
			for (unsigned int i : deform_graph.indices_blocks[block_idx]){
				center += deform_graph.nodes[i].Position;
				count += 1;
			}
		}
	}
	center = center / count;

	for (int block_idx = 0; block_idx < deform_graph.indices_blocks.size(); block_idx++){
		if (blocks_type[block_idx] == 1){
			for (unsigned int i : deform_graph.indices_blocks[block_idx]){
				aim_positions[i] = center + s*(aim_positions[i] - center);
			}
		}
	}
}


void sibr::GaussianView::UpdatePosition(const DeformGraph &dm){
	auto start = std::chrono::steady_clock::now();

	if (! _only_deform_gs){
		#pragma omp parallel for
		for (unsigned int i = 0; i < mesh_vertices.size(); i++){
			if (_optimize_from_start){
				mesh_points[i] = dm.predict_mesh(mesh_vertices[i], cand_points_backup, mesh_points_backup[i]);
			}
			else{
				mesh_points[i] = dm.predict_mesh(mesh_vertices[i], cand_points, mesh_points[i]);	
			}
		}
	}

	#pragma omp parallel for
	for (unsigned int i = 0; i < simplified_vertices.size(); i++){
		if (_optimize_from_start){
			simplified_points[i] = dm.predict_mesh(simplified_vertices[i], cand_points_backup, simplified_points_backup[i]);
		}
		else{
			simplified_points[i] = dm.predict_mesh(simplified_vertices[i], cand_points, simplified_points[i]);	
		}
	}

	#pragma omp parallel for
	for (unsigned int i = 0; i < soup_vertices.size(); i++){
		if (_optimize_from_start){
			soup_points[i] = dm.predict_mesh(soup_vertices[i], cand_points_backup, soup_points_backup[i]);
		}
		else{
			soup_points[i] = dm.predict_mesh(soup_vertices[i], cand_points, soup_points[i]);	
		}
	}

	#pragma omp parallel for
	for (unsigned int i = 0; i < ends_vertices.size(); i++){
		int gs_idx = i / 6;
		if (_optimize_from_start){
			if ((i % 2) == 0){
				ends_vector[gs_idx].ends[(i % 6) / 2].first = dm.predict_mesh(ends_vertices[i], cand_points_backup, ends_vector_backup[gs_idx].ends[(i % 6) / 2].first);
			}
			else{
				ends_vector[gs_idx].ends[(i % 6) / 2].second = dm.predict_mesh(ends_vertices[i], cand_points_backup, ends_vector_backup[gs_idx].ends[(i % 6) / 2].second);
			}
		}
		else{
			if ((i % 2) == 0){
				ends_vector[gs_idx].ends[(i % 6) / 2].first = dm.predict_mesh(ends_vertices[i], cand_points, ends_vector[gs_idx].ends[(i % 6) / 2].first);
			}
			else{
				ends_vector[gs_idx].ends[(i % 6) / 2].second = dm.predict_mesh(ends_vertices[i], cand_points, ends_vector[gs_idx].ends[(i % 6) / 2].second);
			}	
		}
	}

	std::vector<Node> next_nodes = deform_graph.nodes;
	#pragma omp parallel for
	for (unsigned int node_idx = 0; node_idx < deform_graph.nodes.size(); node_idx++){
		unsigned int i = deform_graph.nodes[node_idx].Vertex_index;
		next_nodes[node_idx].Position = dm.predict_mesh(cand_vertices[i], cand_points, next_nodes[node_idx].Position);
	}

	#pragma omp parallel for
	for (unsigned int node_idx = 0; node_idx < deform_graph.nodes.size(); node_idx++){
		deform_graph.nodes[node_idx].Position = next_nodes[node_idx].Position;
	}

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsedSeconds = end - start;

}


void sibr::GaussianView::UpdatePositionforSamples(const DeformGraph &dm){
	auto start = std::chrono::steady_clock::now();

	#pragma omp parallel for
	for (unsigned int i = 0; i < sample_positions.size(); i++){
		if (static_samples[i]) {continue;}

		if (_optimize_from_start){
			sample_positions[i] = dm.predict_samples(i, sps.sample_points[i], cand_points_backup, backup_sample_positions);
		}
		else {
			sample_positions[i] = dm.predict_samples(i, sps.sample_points[i], cand_points, sample_positions);
		}
	}

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsedSeconds = end - start;
}



void sibr::GaussianView::UpdateAsSixPointsWithdrawBad(const DeformGraph &dm){
	auto start = std::chrono::steady_clock::now();

	#pragma omp parallel for
	for (int gs_idx = 0; gs_idx < count; gs_idx++){
		if (static_gaussians[gs_idx]) {continue;}

		Rot o_rot = rot_vector[gs_idx];
		Eigen::Quaternionf origin_q(o_rot.rot[0], o_rot.rot[1], o_rot.rot[2], o_rot.rot[3]);

		Eigen::Matrix<float, 3, 6> Q;
		Eigen::Matrix<float, 3, 6> P;

		for (int i = 0; i < 3; i++){
			Eigen::Vector3f zeros_v(0.0f, 0.0f, 0.0f);
			Eigen::Vector3f v3;
			v3 = zeros_v;
			v3(i) = 1.0f;
			Q.col(i*2) = v3;
			v3 = zeros_v;
			v3(i) = -1.0f;
			Q.col(i*2 + 1) = v3;

			v3 << ends_vector[gs_idx].ends[i].first;
			P.col(i*2) = v3;
			v3 << ends_vector[gs_idx].ends[i].second;
			P.col(i*2 + 1) = v3;
		}

		Pos center = P.rowwise().mean();
		P.colwise() -= center;

		Eigen::Matrix3f M = P * Q.completeOrthogonalDecomposition().pseudoInverse();

		Eigen::Vector3f K;
		
		Eigen::Matrix3f dest_rot_o = getOthogonalMatrixWithK(M, K);

		Eigen::Quaternionf q(dest_rot_o);
		q.normalize();

		// Update Rotations
		rot_vector[gs_idx] = {q.w(), q.x(), q.y(), q.z()};

		// Update Scales
		for (int i = 0; i < 3; i++) {
			scale_vector[gs_idx].scale[i] = K(i)/ ((scale_backup_vector[gs_idx].scale[i]+axis_padding)*end_coeff)*scale_backup_vector[gs_idx].scale[i];
		}

		// Update Center
		pos_vector[gs_idx] = center;

		// Update SHs
		Eigen::Matrix3f shs_rot = (q*(origin_q.inverse())).normalized().toRotationMatrix();

		std::vector<float> shs_temp(shs_vector[gs_idx].shs, shs_vector[gs_idx].shs + 48);

		for (int k = 0; k < 16; k++){
			if (k % 2 == 1){
				shs_temp[k*3] = -shs_temp[k*3];
				shs_temp[k*3 + 1] = -shs_temp[k*3 + 1];
				shs_temp[k*3 + 2] = -shs_temp[k*3 + 2];
			}
		}

		SH_Rotation(shs_rot, shs_temp);

		for (int k = 0; k < 16; k++){
			if (k % 2 == 1){
				shs_temp[k*3] = -shs_temp[k*3];
				shs_temp[k*3 + 1] = -shs_temp[k*3 + 1];
				shs_temp[k*3 + 2] = -shs_temp[k*3 + 2];
			}
		}

		std::copy(shs_temp.data(), shs_temp.data() + 48, shs_vector[gs_idx].shs);
	}

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsedSeconds = end - start;

	if (!_only_deform_gs){
		FastUpdateSamplesSH(dm);
	}
	return;
}


void sibr::GaussianView::FastUpdateSamplesSH(const DeformGraph &dm){
	vector<Eigen::Quaternionf> q_vector(dm.nodes.size());

	#pragma omp parallel for
	for (unsigned int i = 0; i < dm.nodes.size(); i++) {
		array<double, 9> rotation = dm.rot[i];
		Eigen::Matrix3f othomat = FastgetOthogonalMatrix(CastRot2Matrix(rotation));

		Eigen::Quaternionf q(othomat);
		q.normalize();
		q_vector[i] = q;
	}

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(q_vector_cuda, q_vector.data(), sizeof(Eigen::Quaternionf) * q_vector.size(), cudaMemcpyHostToDevice));

	RotateSamplesSHsCUDA(sample_positions.size(), sample_neighbor_weights, sample_neighbor_nodes, q_vector_cuda, aim_feature_cuda, dm.k_nearest, static_samples_cuda);

}



void sibr::GaussianView::RenderHelpers(const sibr::Viewport& viewport){

	if (!shadersCompiled) {
		initGraphShader();
		initgsHelperShader();
		initMeshShader();
	}


	if (flag_rect){
		GLboolean blendState;
		glGetBooleanv(GL_BLEND, &blendState);
		GLint blendSrc, blendDst;
		glGetIntegerv(GL_BLEND_SRC_ALPHA, &blendSrc);
		glGetIntegerv(GL_BLEND_DST_ALPHA, &blendDst);

		// Enable basic blending.
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_BLEND);

		viewport.bind();
		gsHelperShader.begin();
		ratiogsHelper2Dgpu.set(1.0f);
		gsHelperStateGPU.set(1);
		getQuadMesh();
		quadMesh->render_lines();
		gsHelperShader.end();

		if (!blendState) {
			glDisable(GL_BLEND);
		}
		glBlendFunc(blendSrc, blendDst);
	}

	if(flag_show_nodes) {
		viewport.bind();
		GraphShader.begin();
		PVM.set(T);
		getGraphMesh();
		graphMesh->render_lines();
		GraphShader.end();
	}

	if(flag_show_samples) {
		glClear(GL_DEPTH_BUFFER_BIT);
		if (sps.sample_points.size() != 0) {
			viewport.bind();
			GraphShader.begin();
			PVM.set(T);
			samplesMesh->render_points(5.0f);
			GraphShader.end();
		}
	}
}


void sibr::GaussianView::getGraphMesh(){
	graphMesh = std::shared_ptr<Mesh>(new Mesh(true));

	std::vector<float> vertexBuffer;
	std::vector<sibr::Vector2f> alphaBuffer;
	for (int i = 0; i < deform_graph.nodes.size(); i++){
		for (int j = 0; j < 3; j++){
			vertexBuffer.push_back(deform_graph.nodes[i].Position(j));
		}
	}

	std::vector<uint> indicesBuffer = deform_graph.indices;

	// HACK: align for the shit data structure of indices in SIBR
	if (indicesBuffer.size() % 3 == 1){
		indicesBuffer.push_back(indicesBuffer[indicesBuffer.size() - 1]);
	}
	if (indicesBuffer.size() % 3 == 2){
		indicesBuffer.push_back(indicesBuffer[indicesBuffer.size() - 1]);
	}
	// END HACK

	std::vector<sibr::Vector3f> colorsBuffer;

	for (int i = 0; i < deform_graph.nodes.size(); i++){
		bool found = false;
		for (int block_idx = 0; block_idx < deform_graph.indices_blocks.size(); block_idx++){
			auto it = std::find(deform_graph.indices_blocks[block_idx].begin(), deform_graph.indices_blocks[block_idx].end(), i);
			if (it != deform_graph.indices_blocks[block_idx].end()){
				found = true;
				if (blocks_type[block_idx] == 1){
					colorsBuffer.push_back(sibr::Vector3f(0.2, 0.9, 0.2));
				}
				else if (blocks_type[block_idx] == 0){
					colorsBuffer.push_back(sibr::Vector3f(0.8, 0.8, 0.2));
				}
				else{
					colorsBuffer.push_back(sibr::Vector3f(0.9, 0.2, 0.2));
				}
				break;
			}

		}
		if (!found) {
			colorsBuffer.push_back(sibr::Vector3f(0.6, 0.6, 0.6));
		}
		alphaBuffer.push_back(sibr::Vector2f(1.0, 1.0));
	}

	graphMesh->colors(colorsBuffer);
	graphMesh->vertices(vertexBuffer);
	graphMesh->triangles(indicesBuffer);
	graphMesh->texCoords(alphaBuffer);
}


void sibr::GaussianView::getCentersMesh(){
	centersMesh = std::shared_ptr<Mesh>(new Mesh(true));

	std::vector<float> vertexBuffer(aim_centers.size()*v_sphere.size()*3);
	std::vector<sibr::Vector3f> colorsBuffer(aim_centers.size()*v_sphere.size());
	std::vector<sibr::Vector2f> alphaBuffer(aim_centers.size()*v_sphere.size());
	std::vector<uint> indicesBuffer(aim_centers.size()*f_sphere.size()*3);

	
	for (int i = 0; i < aim_centers.size(); i++) {

		#pragma omp parallel for
		for (int k = 0; k < v_sphere.size(); k++){
			int v_offset = i*v_sphere.size() + k;

			float r = aim_centers_radius[i];

			sibr::Vector3f xyz =  sibr::Vector3f(v_sphere[k].x*r, v_sphere[k].y*r, v_sphere[k].z*r);

			vertexBuffer[v_offset*3 + 0] = (xyz.x() + aim_centers[i].x());
			vertexBuffer[v_offset*3 + 1] = (xyz.y() + aim_centers[i].y());
			vertexBuffer[v_offset*3 + 2] = (xyz.z() + aim_centers[i].z());
			
			if (i == deform_graph.current_control_idx){
				if (! _during_deform){
					colorsBuffer[v_offset] = (sibr::Vector3f(0.1f, 0.6f, 0.1f));
				}
				else{
					colorsBuffer[v_offset] = (sibr::Vector3f(0.2f, 1.0f, 0.2f));
				}
				alphaBuffer[v_offset] = (sibr::Vector2f(0.3, 1.0));
			}
			else {
				colorsBuffer[v_offset] = (sibr::Vector3f(0.8f, 0.1f, 0.1f));
				alphaBuffer[v_offset] = (sibr::Vector2f(0.4, 1.0));
			}
		}

		#pragma omp parallel for
		for (int k = 0; k < f_sphere.size(); k++){
			int f_offset = i*f_sphere.size() + k;
			indicesBuffer[f_offset*3 + 0] = (v_sphere.size()*i + f_sphere[k].v1 - 1);
			indicesBuffer[f_offset*3 + 1] = (v_sphere.size()*i + f_sphere[k].v2 - 1);
			indicesBuffer[f_offset*3 + 2] = (v_sphere.size()*i + f_sphere[k].v3 - 1);
		}

	}
	

	centersMesh->colors(colorsBuffer);
	centersMesh->vertices(vertexBuffer);
	centersMesh->triangles(indicesBuffer);
	centersMesh->texCoords(alphaBuffer);
}

void sibr::GaussianView::getSamplesMesh(){
	samplesMesh = std::shared_ptr<Mesh>(new Mesh(true));

	std::vector<float> vertexBuffer;
	std::vector<sibr::Vector3f> colorsBuffer;
	std::vector<sibr::Vector2f> alphaBuffer;
	std::vector<uint> indicesBuffer;
	for (int i = 0; i < sample_positions.size(); i++){
		if (! show_aim){
			float alpha = feature_opacity_vector[i];
			if (alpha > _show_alpha_cutoff){
				for (int j = 0; j < 3; j++){
					vertexBuffer.push_back(sample_positions[i](j));
				}

				float r = min(1.0, max(0.0, (SH0*sample_feature_shs[i].shs[0]/alpha + 0.5)));
				float g = min(1.0, max(0.0, (SH0*sample_feature_shs[i].shs[1]/alpha + 0.5)));
				float b = min(1.0, max(0.0, (SH0*sample_feature_shs[i].shs[2]/alpha + 0.5)));


				colorsBuffer.push_back(sibr::Vector3f(r, g, b));
				if (_enable_alpha){
					alphaBuffer.push_back(sibr::Vector2f(min(1.0f, alpha), 1.0));
				}
				else{
					alphaBuffer.push_back(sibr::Vector2f(min(1.0f, 1.0f), 1.0));
				}
				indicesBuffer.push_back((uint)i);
			}
		}
		else if (! show_error){
			int orig_sample_idx = i;
			if((aim_feature_shs[orig_sample_idx].shs[0] != 0.0) || (aim_feature_shs[orig_sample_idx].shs[1] != 0.0) || (aim_feature_shs[orig_sample_idx].shs[2] != 0.0)){
				float alpha = aim_opacity[orig_sample_idx] + 1e-4;
				if (alpha <= _show_alpha_cutoff){continue;}

				for (int j = 0; j < 3; j++){
					vertexBuffer.push_back(sample_positions[i](j));
				}

				float r = min(1.0, max(0.0, (SH0*aim_feature_shs[orig_sample_idx].shs[0]/alpha + 0.5)));
				float g = min(1.0, max(0.0, (SH0*aim_feature_shs[orig_sample_idx].shs[1]/alpha + 0.5)));
				float b = min(1.0, max(0.0, (SH0*aim_feature_shs[orig_sample_idx].shs[2]/alpha + 0.5)));

				colorsBuffer.push_back(sibr::Vector3f(r, g, b));
				if (_enable_alpha){
					alphaBuffer.push_back(sibr::Vector2f(min(1.0f, alpha), 1.0));
				}
				else{
					alphaBuffer.push_back(sibr::Vector2f(min(1.0f, 1.0f), 1.0));
				}
				indicesBuffer.push_back((uint)i);
			}
		}
		else if (! show_all_samples){
			int orig_sample_idx = i;
			if((aim_feature_shs[orig_sample_idx].shs[0]!= 0.0)||( sample_feature_shs[i].shs[0] != 0.0) || (aim_feature_shs[orig_sample_idx].shs[1]!= 0.0)||( sample_feature_shs[i].shs[1]!= 0.0) || (aim_feature_shs[orig_sample_idx].shs[2] != 0.0)||( sample_feature_shs[i].shs[2] != 0.0)){

				float r0, g0, b0, r, g, b;
				r0 = 0.0f;
				r0 += SH0*abs(aim_feature_shs[orig_sample_idx].shs[0]*aim_opacity[orig_sample_idx]- sample_feature_shs[i].shs[0]*feature_opacity_vector[i]);
				r0 += SH0*abs(aim_feature_shs[orig_sample_idx].shs[1]*aim_opacity[orig_sample_idx]- sample_feature_shs[i].shs[1]*feature_opacity_vector[i]);
				r0 += SH0*abs(aim_feature_shs[orig_sample_idx].shs[2]*aim_opacity[orig_sample_idx]- sample_feature_shs[i].shs[2]*feature_opacity_vector[i]);

				if (r0 < _show_alpha_cutoff){continue;}

				r = min(1.0f, r0*20.0f);
				g = 0.0f;
				b = 1-r;

				float color_max = max(r, g);
				color_max = max(color_max, b);

				for (int j = 0; j < 3; j++){
					vertexBuffer.push_back(sample_positions[i](j));
				}

				colorsBuffer.push_back(sibr::Vector3f(r, g, b));
				alphaBuffer.push_back(sibr::Vector2f(1.0, 1.0));
				indicesBuffer.push_back((uint)i);
			}
		}
		else {
			int orig_sample_idx = i;
			if (true){
				for (int j = 0; j < 3; j++){
					vertexBuffer.push_back(sample_positions[i](j));
				}

				float r0, g0, b0, r, g, b;
				r0 = 0.0f;
				r0 += SH0*abs(aim_feature_shs[orig_sample_idx].shs[0]- sample_feature_shs[i].shs[0]);
				r0 += SH0*abs(aim_feature_shs[orig_sample_idx].shs[1]- sample_feature_shs[i].shs[1]);
				r0 += SH0*abs(aim_feature_shs[orig_sample_idx].shs[2]- sample_feature_shs[i].shs[2]);

				r = min(1.0f, r0*20.0f);
				g = 0.0f;
				b = 1-r;

				float color_max = max(r, g);
				color_max = max(color_max, b);

				colorsBuffer.push_back(sibr::Vector3f(r, g, b));
				alphaBuffer.push_back(sibr::Vector2f(1.0, 1.0));
				indicesBuffer.push_back((uint)i);
			}
		}
	}
	
	
	// HACK: align for the shit data structure of indices in SIBR
	if (indicesBuffer.size() % 3 == 1){
		indicesBuffer.push_back(indicesBuffer[indicesBuffer.size() - 1]);
	}
	if (indicesBuffer.size() % 3 == 2){
		indicesBuffer.push_back(indicesBuffer[indicesBuffer.size() - 1]);
	}

	samplesMesh->colors(colorsBuffer);
	samplesMesh->vertices(vertexBuffer);
	samplesMesh->triangles(indicesBuffer);
	samplesMesh->texCoords(alphaBuffer);
}


void sibr::GaussianView::getQuadMesh(){
	quadMesh = std::shared_ptr<Mesh>(new Mesh(true));

	float l, r, u ,b;
	l = min(press_x, lastX);
	r = max(press_x, lastX);
	u = window_size(1) - min(press_y, lastY);
	b = window_size(1) - max(press_y, lastY);

	float corners[4][2] = { {(l / window_size(0))*2 -1, (u / window_size(1))*2 -1},		\
							 {(l / window_size(0))*2 -1, (b / window_size(1))*2 -1}, 	\
							 {(r / window_size(0))*2 -1, (b / window_size(1))*2 -1}, 	\
							 {(r / window_size(0))*2 -1, (u / window_size(1))*2 -1} };

	std::vector<float> vertexBuffer;
	for (int i = 0; i < 4; i++) {
		Vector3f corner((float)corners[i][0], (float)corners[i][1], 0.0f);
		for (int c = 0; c < 3; c++) {
			vertexBuffer.push_back(corner[c]);
		}
	}

	int indices[9] = { 0, 1, 1, 2, 2, 3, 3, 0, 0};
	std::vector<uint> indicesBuffer(&indices[0], &indices[0] + 9);

	quadMesh->vertices(vertexBuffer);
	quadMesh->triangles(indicesBuffer);
}

void sibr::GaussianView::onRender(const Viewport & vpRender){

}

void sibr::GaussianView::initgsHelperShader(void)
{

	std::string trackBallVertexShader =
		"#version 450										\n"
		"layout(location = 0) in vec3 in_vertex;			\n"
		"void main(void) {									\n"
		"	gl_Position = vec4(in_vertex.xy,0.0, 1.0);		\n"
		"}													\n";

	std::string trackBallFragmentShader =
		"#version 450														\n"
		"uniform float ratio;												\n"
		"uniform int mState;												\n"
		"out vec4 out_color;												\n"
		"void main(void) {													\n"
		"	out_color = vec4(1.0, 1.0, 0.0, 1.0);							\n"
		"}																	\n";

	gsHelperShader.init("gsHelperShader", trackBallVertexShader, trackBallFragmentShader);

	ratiogsHelper2Dgpu.init(gsHelperShader, "ratio");
	gsHelperStateGPU.init(gsHelperShader, "mState");

	shadersCompiled = true;
}

void sibr::GaussianView::initGraphShader(void){
	std::string GraphVertexShader =
		"#version 450										\n"
		"layout(location = 0) in vec3 in_vertex;			\n"
		"layout(location = 1) in vec3 in_colors;			\n"
		"layout(location = 2) in vec2 in_alpha;				\n"
		"layout (location = 1) out vec4 colors;				\n"
		"uniform mat4 PVM;									\n"
		"void main(void) {									\n"
		"	gl_Position = PVM * vec4(in_vertex, 1.0);		\n"
		"	colors= vec4(in_colors, in_alpha.x);			\n"
		"}													\n";

	std::string GraphFragmentShader =
		"#version 450										\n"
		"uniform float ratio;								\n"
		"uniform int mState;								\n"
		"layout (location = 1) in vec4 colors;				\n"
		"out vec4 out_color;								\n"
		"void main(void) {									\n"
		"	out_color = colors;								\n"
		"}													\n";

	GraphShader.init("GraphShader", GraphVertexShader, GraphFragmentShader);
	PVM.init(GraphShader, "PVM");
}

void sibr::GaussianView::initMeshShader(void){
	std::string MeshVertexShader =
		"#version 450										\n"
		"layout(location = 0) in vec3 in_vertex;			\n"
		"layout(location = 1) in vec3 in_colors;			\n"
		"layout(location = 2) in vec2 in_uvCoords;			\n"

		"layout (location = 2) out vec2 uvCoords;			\n"
		"layout (location = 1) out vec3 colors;				\n"
		"uniform mat4 MVP;									\n"
		"void main(void) {									\n"
		"	uvCoords = in_uvCoords;		\n"
		"	colors= in_colors;		\n"
		"	gl_Position = MVP*vec4(in_vertex,1) ;		\n"
		"}													\n";

	std::string MeshFragmentShader =
		"#version 450														\n"
		"layout(binding = 0) uniform sampler2D tex;				\n"
		"layout (location = 2) in vec2 uvCoords;								\n"
		"layout (location = 1) in vec3 colors;									\n"
		"out vec4 out_color;												\n"
		"void main(void) {													\n"
		"	out_color = texture(tex,vec2(uvCoords.x,1.0-uvCoords.y));\n"
		"}																	\n";

	MeshShader.init("MeshShader", MeshVertexShader, MeshFragmentShader);
	MVP.init(MeshShader, "MVP");
	Tex.init(MeshShader, "tex");
}


void sibr::GaussianView::getOverallAABB(){
	auto start_aabb = std::chrono::steady_clock::now();

	aabb_overall.xyz_min = pos_vector[0];
	aabb_overall.xyz_max = pos_vector[0];
	for (int i = 0; i < count; i++){
		aabb_overall.xyz_min.x() = min(aabb_overall.xyz_min.x(), pos_vector[i].x());
		aabb_overall.xyz_min.y() = min(aabb_overall.xyz_min.y(), pos_vector[i].y());
		aabb_overall.xyz_min.z() = min(aabb_overall.xyz_min.z(), pos_vector[i].z());
		
		aabb_overall.xyz_max.x() = max(aabb_overall.xyz_max.x(), pos_vector[i].x());
		aabb_overall.xyz_max.y() = max(aabb_overall.xyz_max.y(), pos_vector[i].y());
		aabb_overall.xyz_max.z() = max(aabb_overall.xyz_max.z(), pos_vector[i].z());
	}
	Pos xyz_mean = (aabb_overall.xyz_max + aabb_overall.xyz_min) / 2.0;

	aabb_overall.xyz_min = (aabb_overall.xyz_min - xyz_mean) * 1.1 + xyz_mean;
	aabb_overall.xyz_max = ((aabb_overall.xyz_max - xyz_mean) * 1.1 + xyz_mean - aabb_overall.xyz_min) * (1.0 + (1.0)/NUM_SAMPLES_PER_DIM) + aabb_overall.xyz_min;


	aabb_overall.xyz_min.x() = min(aabb_overall.xyz_min.x(), -0.75f);
	aabb_overall.xyz_min.y() = min(aabb_overall.xyz_min.y(), -0.75f);
	aabb_overall.xyz_min.z() = min(aabb_overall.xyz_min.z(), -0.75f);
	aabb_overall.xyz_max.x() = max(aabb_overall.xyz_max.x(), 0.75f);
	aabb_overall.xyz_max.y() = max(aabb_overall.xyz_max.y(), 0.75f);
	aabb_overall.xyz_max.z() = max(aabb_overall.xyz_max.z(), 0.75f);

	auto end_aabb = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsedSeconds_aabb = end_aabb - start_aabb;
	std::cout << "Overall AABB took " << elapsedSeconds_aabb.count() << " seconds." << std::endl;
}


void sibr::GaussianView::UpdateContainingRelationship(){

	getOverallAABB();

	x_step = (aabb_overall.xyz_max.x() - aabb_overall.xyz_min.x()) / grid_num;
	y_step = (aabb_overall.xyz_max.y() - aabb_overall.xyz_min.y()) / grid_num;
	z_step = (aabb_overall.xyz_max.z() - aabb_overall.xyz_min.z()) / grid_num;

	grid_step = max(max(x_step, y_step), z_step);

	min_xyz = {aabb_overall.xyz_min.x(), aabb_overall.xyz_min.y(), aabb_overall.xyz_min.z()};

	CUDA_SAFE_CALL_ALWAYS(cudaMemset(grid_gs_prefix_sum_cuda, 0, sizeof(int) * grid_count));

	vector<AABB> gs_aabbs(count);
	float* gs_aabbs_cuda;
	#pragma omp parallel for
	for (int gs_idx = 0; gs_idx < count; gs_idx++){
		Eigen::Quaternionf origin_q(rot_vector[gs_idx].rot[0], rot_vector[gs_idx].rot[1], rot_vector[gs_idx].rot[2], rot_vector[gs_idx].rot[3]);
		Eigen::Matrix3f rotation = origin_q.normalized().toRotationMatrix();
		gs_aabbs[gs_idx].xyz_max = pos_vector[gs_idx];
		gs_aabbs[gs_idx].xyz_min = pos_vector[gs_idx];

		Scale s_3d;

		if (opacity_vector[gs_idx] <= CUTOFF_ALPHA){
			s_3d.scale[0] = 0.0f; 
			s_3d.scale[1] = 0.0f; 
			s_3d.scale[2] = 0.0f;
		}
		else{
			s_3d.scale[0] = sqrt(-2.0f*log(CUTOFF_ALPHA/opacity_vector[gs_idx])) * (scale_vector[gs_idx].scale[0]+0.0f);
			s_3d.scale[1] = sqrt(-2.0f*log(CUTOFF_ALPHA/opacity_vector[gs_idx])) * (scale_vector[gs_idx].scale[1]+0.0f);
			s_3d.scale[2] = sqrt(-2.0f*log(CUTOFF_ALPHA/opacity_vector[gs_idx])) * (scale_vector[gs_idx].scale[2]+0.0f);
		}

		scale_3d_clip[gs_idx] = s_3d;

		for (int dim = 0; dim < 3; dim++) {
			Pos sample_local_left(0.0f, 0.0f, 0.0f);
			sample_local_left(dim) = s_3d.scale[dim]; // Edit
			sample_local_left = rotation * sample_local_left;
			Pos sample_local_right = -sample_local_left;
			sample_local_left = (pos_vector[gs_idx] + sample_local_left);
			sample_local_right = (pos_vector[gs_idx] + sample_local_right);

			gs_aabbs[gs_idx].xyz_min.x() = min(gs_aabbs[gs_idx].xyz_min.x(), sample_local_left.x());
			gs_aabbs[gs_idx].xyz_min.y() = min(gs_aabbs[gs_idx].xyz_min.y(), sample_local_left.y());
			gs_aabbs[gs_idx].xyz_min.z() = min(gs_aabbs[gs_idx].xyz_min.z(), sample_local_left.z());
			gs_aabbs[gs_idx].xyz_max.x() = max(gs_aabbs[gs_idx].xyz_max.x(), sample_local_left.x());
			gs_aabbs[gs_idx].xyz_max.y() = max(gs_aabbs[gs_idx].xyz_max.y(), sample_local_left.y());
			gs_aabbs[gs_idx].xyz_max.z() = max(gs_aabbs[gs_idx].xyz_max.z(), sample_local_left.z());

			gs_aabbs[gs_idx].xyz_min.x() = min(gs_aabbs[gs_idx].xyz_min.x(), sample_local_right.x());
			gs_aabbs[gs_idx].xyz_min.y() = min(gs_aabbs[gs_idx].xyz_min.y(), sample_local_right.y());
			gs_aabbs[gs_idx].xyz_min.z() = min(gs_aabbs[gs_idx].xyz_min.z(), sample_local_right.z());
			gs_aabbs[gs_idx].xyz_max.x() = max(gs_aabbs[gs_idx].xyz_max.x(), sample_local_right.x());
			gs_aabbs[gs_idx].xyz_max.y() = max(gs_aabbs[gs_idx].xyz_max.y(), sample_local_right.y());
			gs_aabbs[gs_idx].xyz_max.z() = max(gs_aabbs[gs_idx].xyz_max.z(), sample_local_right.z());
		}
	}

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&gs_aabbs_cuda, sizeof(AABB) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(gs_aabbs_cuda, gs_aabbs.data(), count*sizeof(AABB), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos_vector.data(), sizeof(Pos) * count, cudaMemcpyHostToDevice));

	GetBoxedGsNumPerGridCUDA(count, grid_step, grid_num, gs_aabbs_cuda, grid_gs_prefix_sum_cuda, min_xyz, padding);

	CUDA_SAFE_CALL_ALWAYS(cudaFree(gs_aabbs_cuda));

	grid_gs_prefix_sum_vector.resize(grid_count);
	
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(grid_gs_prefix_sum_vector.data(), grid_gs_prefix_sum_cuda, grid_count*sizeof(int), cudaMemcpyDeviceToHost));


	grided_gs_idx.resize(grid_gs_prefix_sum_vector[grid_count-1]);
	vector<int> grid_used_gs(grid_count);
	grid_used_gs.assign(grid_count, 0);

	// #pragma omp parallel for
	for (int i = 0; i < count; i++){
		int x_idx_min = max(int(floor((gs_aabbs[i].xyz_min.x() - min_xyz.x)/grid_step) - padding), 0);
		int y_idx_min = max(int(floor((gs_aabbs[i].xyz_min.y() - min_xyz.y)/grid_step) - padding), 0);
		int z_idx_min = max(int(floor((gs_aabbs[i].xyz_min.z() - min_xyz.z)/grid_step) - padding), 0);
		
		int x_idx_max = min(int(floor((gs_aabbs[i].xyz_max.x() - min_xyz.x)/grid_step) + padding), grid_num-1);
		int y_idx_max = min(int(floor((gs_aabbs[i].xyz_max.y() - min_xyz.y)/grid_step) + padding), grid_num-1);
		int z_idx_max = min(int(floor((gs_aabbs[i].xyz_max.z() - min_xyz.z)/grid_step) + padding), grid_num-1);

		for (int x_idx = x_idx_min; x_idx < x_idx_max + 1; x_idx++){
			for (int y_idx = y_idx_min; y_idx < y_idx_max + 1; y_idx++){
				for (int z_idx = z_idx_min; z_idx < z_idx_max + 1; z_idx++){
					int grid_idx = x_idx*grid_num*grid_num + y_idx*grid_num + z_idx;
					if (grid_idx == 0){
						grided_gs_idx[grid_used_gs[grid_idx]] = i;
					}
					else {
						grided_gs_idx[grid_gs_prefix_sum_vector[grid_idx-1] + grid_used_gs[grid_idx]] = i;
					}
					grid_used_gs[grid_idx] += 1;
				}
			}
		}
	}

	CUDA_SAFE_CALL_ALWAYS(cudaFree(grided_gs_idx_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&grided_gs_idx_cuda, grid_gs_prefix_sum_vector[grid_count-1]*sizeof(int)));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(grided_gs_idx_cuda, grided_gs_idx.data(), grid_gs_prefix_sum_vector[grid_count-1]*sizeof(int), cudaMemcpyHostToDevice));

}


void sibr::GaussianView::UpdateContainingRelationshipWithStaticGrids(){
	auto start_update = std::chrono::steady_clock::now();

	getOverallAABB();

	x_step = (aabb_overall.xyz_max.x() - aabb_overall.xyz_min.x()) / grid_num;
	y_step = (aabb_overall.xyz_max.y() - aabb_overall.xyz_min.y()) / grid_num;
	z_step = (aabb_overall.xyz_max.z() - aabb_overall.xyz_min.z()) / grid_num;

	grid_step = max(max(x_step, y_step), z_step);

	min_xyz = {aabb_overall.xyz_min.x(), aabb_overall.xyz_min.y(), aabb_overall.xyz_min.z()};

	CUDA_SAFE_CALL_ALWAYS(cudaMemset(grid_gs_prefix_sum_cuda, 0, sizeof(int) * grid_count));

	vector<AABB> gs_aabbs(count);
	float* gs_aabbs_cuda;
	#pragma omp parallel for
	for (int gs_idx = 0; gs_idx < count; gs_idx++){
		Eigen::Quaternionf origin_q(rot_vector[gs_idx].rot[0], rot_vector[gs_idx].rot[1], rot_vector[gs_idx].rot[2], rot_vector[gs_idx].rot[3]);
		Eigen::Matrix3f rotation = origin_q.normalized().toRotationMatrix();
		gs_aabbs[gs_idx].xyz_max = pos_vector[gs_idx];
		gs_aabbs[gs_idx].xyz_min = pos_vector[gs_idx];

		Scale s_3d;

		if (opacity_vector[gs_idx] <= CUTOFF_ALPHA){
			s_3d.scale[0] = 0.0f; 
			s_3d.scale[1] = 0.0f; 
			s_3d.scale[2] = 0.0f;
		}
		else{
			s_3d.scale[0] = sqrt(-2.0f*log(CUTOFF_ALPHA/opacity_vector[gs_idx])) * (scale_vector[gs_idx].scale[0]+0.0f);
			s_3d.scale[1] = sqrt(-2.0f*log(CUTOFF_ALPHA/opacity_vector[gs_idx])) * (scale_vector[gs_idx].scale[1]+0.0f);
			s_3d.scale[2] = sqrt(-2.0f*log(CUTOFF_ALPHA/opacity_vector[gs_idx])) * (scale_vector[gs_idx].scale[2]+0.0f);
		}

		scale_3d_clip[gs_idx] = s_3d;

		for (int dim = 0; dim < 3; dim++) {
			Pos sample_local_left(0.0f, 0.0f, 0.0f);
			sample_local_left(dim) = s_3d.scale[dim]; // Edit
			sample_local_left = rotation * sample_local_left;
			Pos sample_local_right = -sample_local_left;
			sample_local_left = (pos_vector[gs_idx] + sample_local_left);
			sample_local_right = (pos_vector[gs_idx] + sample_local_right);

			gs_aabbs[gs_idx].xyz_min.x() = min(gs_aabbs[gs_idx].xyz_min.x(), sample_local_left.x());
			gs_aabbs[gs_idx].xyz_min.y() = min(gs_aabbs[gs_idx].xyz_min.y(), sample_local_left.y());
			gs_aabbs[gs_idx].xyz_min.z() = min(gs_aabbs[gs_idx].xyz_min.z(), sample_local_left.z());
			gs_aabbs[gs_idx].xyz_max.x() = max(gs_aabbs[gs_idx].xyz_max.x(), sample_local_left.x());
			gs_aabbs[gs_idx].xyz_max.y() = max(gs_aabbs[gs_idx].xyz_max.y(), sample_local_left.y());
			gs_aabbs[gs_idx].xyz_max.z() = max(gs_aabbs[gs_idx].xyz_max.z(), sample_local_left.z());

			gs_aabbs[gs_idx].xyz_min.x() = min(gs_aabbs[gs_idx].xyz_min.x(), sample_local_right.x());
			gs_aabbs[gs_idx].xyz_min.y() = min(gs_aabbs[gs_idx].xyz_min.y(), sample_local_right.y());
			gs_aabbs[gs_idx].xyz_min.z() = min(gs_aabbs[gs_idx].xyz_min.z(), sample_local_right.z());
			gs_aabbs[gs_idx].xyz_max.x() = max(gs_aabbs[gs_idx].xyz_max.x(), sample_local_right.x());
			gs_aabbs[gs_idx].xyz_max.y() = max(gs_aabbs[gs_idx].xyz_max.y(), sample_local_right.y());
			gs_aabbs[gs_idx].xyz_max.z() = max(gs_aabbs[gs_idx].xyz_max.z(), sample_local_right.z());
		}
	}

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&gs_aabbs_cuda, sizeof(AABB) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(gs_aabbs_cuda, gs_aabbs.data(), count*sizeof(AABB), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos_vector.data(), sizeof(Pos) * count, cudaMemcpyHostToDevice));

	GetBoxedGsNumPerGridCUDA(count, grid_step, grid_num, gs_aabbs_cuda, grid_gs_prefix_sum_cuda, min_xyz, padding);

	CUDA_SAFE_CALL_ALWAYS(cudaFree(gs_aabbs_cuda));

	grid_gs_prefix_sum_vector.resize(grid_count);
	
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(grid_gs_prefix_sum_vector.data(), grid_gs_prefix_sum_cuda, grid_count*sizeof(int), cudaMemcpyDeviceToHost));


	grided_gs_idx.resize(grid_gs_prefix_sum_vector[grid_count-1]);
	vector<int> grid_used_gs(grid_count);
	grid_used_gs.assign(grid_count, 0);

	current_static_grids.assign(grid_count, 1);

	// #pragma omp parallel for
	for (int i = 0; i < count; i++){
		int x_idx_min = max(int(floor((gs_aabbs[i].xyz_min.x() - min_xyz.x)/grid_step) - padding), 0);
		int y_idx_min = max(int(floor((gs_aabbs[i].xyz_min.y() - min_xyz.y)/grid_step) - padding), 0);
		int z_idx_min = max(int(floor((gs_aabbs[i].xyz_min.z() - min_xyz.z)/grid_step) - padding), 0);
		
		int x_idx_max = min(int(floor((gs_aabbs[i].xyz_max.x() - min_xyz.x)/grid_step) + padding), grid_num-1);
		int y_idx_max = min(int(floor((gs_aabbs[i].xyz_max.y() - min_xyz.y)/grid_step) + padding), grid_num-1);
		int z_idx_max = min(int(floor((gs_aabbs[i].xyz_max.z() - min_xyz.z)/grid_step) + padding), grid_num-1);

		for (int x_idx = x_idx_min; x_idx < x_idx_max + 1; x_idx++){
			for (int y_idx = y_idx_min; y_idx < y_idx_max + 1; y_idx++){
				for (int z_idx = z_idx_min; z_idx < z_idx_max + 1; z_idx++){
					int grid_idx = x_idx*grid_num*grid_num + y_idx*grid_num + z_idx;
					if (grid_idx == 0){
						grided_gs_idx[grid_used_gs[grid_idx]] = i;
					}
					else {
						grided_gs_idx[grid_gs_prefix_sum_vector[grid_idx-1] + grid_used_gs[grid_idx]] = i;
					}
					grid_used_gs[grid_idx] += 1;

					if (moved_gaussians[i] == 1){
						current_static_grids[grid_idx] = 0;
					}
				}
			}
		}
	}

	CUDA_SAFE_CALL_ALWAYS(cudaFree(grided_gs_idx_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&grided_gs_idx_cuda, grid_gs_prefix_sum_vector[grid_count-1]*sizeof(int)));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(grided_gs_idx_cuda, grided_gs_idx.data(), grid_gs_prefix_sum_vector[grid_count-1]*sizeof(int), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(current_static_grids_cuda, current_static_grids.data(), grid_count*sizeof(int), cudaMemcpyHostToDevice));

	auto end_update = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsedSeconds_update = end_update - start_update;
	std::cout << "Update Containing Relationship took " << elapsedSeconds_update.count() << " seconds." << std::endl;
}


std::vector<SamplePoint> sibr::GaussianView::getGridSamples(){
	auto start_bbx = std::chrono::steady_clock::now();

	x_step = (aabb_overall.xyz_max.x() - aabb_overall.xyz_min.x()) / grid_num;
	y_step = (aabb_overall.xyz_max.y() - aabb_overall.xyz_min.y()) / grid_num;
	z_step = (aabb_overall.xyz_max.z() - aabb_overall.xyz_min.z()) / grid_num;

	grid_step = max(max(x_step, y_step), z_step);
	init_grid_step = grid_step;
	init_sample_step = init_grid_step / (float) SAMPLES_PER_GRID;

	low_pass_filter_param = init_sample_step * init_sample_step * lpf_parameter;

	std::cout << "grid step: " << grid_step << std::endl;

	grid_count = grid_num*grid_num*grid_num;
	samples_count_per_grid = SAMPLES_PER_GRID*SAMPLES_PER_GRID*SAMPLES_PER_GRID;
	min_xyz = {aabb_overall.xyz_min.x(), aabb_overall.xyz_min.y(), aabb_overall.xyz_min.z()};

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&grid_gs_prefix_sum_cuda, sizeof(int) * grid_count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(grid_gs_prefix_sum_cuda, 0, sizeof(int) * grid_count));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&current_static_grids_cuda, sizeof(int) * grid_count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(current_static_grids_cuda, 0, sizeof(int) * grid_count));
	current_static_grids = vector<int>(grid_count, 0);

	GetGsNumPerGridCUDA(count, grid_step, grid_num, pos_cuda, grid_gs_prefix_sum_cuda, min_xyz);

	grid_gs_prefix_sum_vector.resize(grid_count);
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(grid_gs_prefix_sum_vector.data(), grid_gs_prefix_sum_cuda, grid_count*sizeof(int), cudaMemcpyDeviceToHost));

	valid_grid_num = 0;
	for (int i = 0; i < grid_count; i++){
		if (i == 0){
			if (grid_gs_prefix_sum_vector[i] != 0){
				valid_grid_num += 1;
				valid_grid_idx.push_back(i);
			}
		}
		else if ((grid_gs_prefix_sum_vector[i]-grid_gs_prefix_sum_vector[i-1]) != 0) {
			valid_grid_num += 1;
			valid_grid_idx.push_back(i);
		}
	}
	std::cout << "there are totally " << valid_grid_num << " grids containing gaussians." << std::endl; 
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&valid_grid_cuda, sizeof(int) * valid_grid_num));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(valid_grid_cuda, valid_grid_idx.data(), valid_grid_num*sizeof(int), cudaMemcpyHostToDevice));

	new_gs_idx.resize(count);
	vector<int> grid_used_gs(grid_count);
	grid_used_gs.assign(grid_count, 0);

	for (int i = 0; i < count; i++){
		int x_idx = int(floor((pos_vector[i].x() - min_xyz.x)/grid_step));
		int y_idx = int(floor((pos_vector[i].y() - min_xyz.y)/grid_step));
		int z_idx = int(floor((pos_vector[i].z() - min_xyz.z)/grid_step));
		int grid_idx = x_idx*grid_num*grid_num + y_idx*grid_num + z_idx;
		if (grid_idx == 0){
			new_gs_idx[i] = grid_used_gs[grid_idx];
		}
		else {
			new_gs_idx[i] = grid_gs_prefix_sum_vector[grid_idx-1] + grid_used_gs[grid_idx];
		}
		grid_used_gs[grid_idx] += 1;
	}


	// Re-Order the gaussians
	for (int i = 0; i < count; i++){
		pos_vector[new_gs_idx[i]] = pos_backup_vector[i];
		rot_vector[new_gs_idx[i]] = rot_backup_vector[i];
		scale_vector[new_gs_idx[i]] = scale_backup_vector[i];
		opacity_vector[new_gs_idx[i]] = opacity_backup_vector[i];
		shs_vector[new_gs_idx[i]] = shs_backup_vector[i];
		vertices[i].index = i;
	}
	
	for (int i = 0; i < count; i++){
		pos_backup_vector[i] = pos_vector[i];
		rot_backup_vector[i] = rot_vector[i];
		scale_backup_vector[i] = scale_vector[i];
		opacity_backup_vector[i] = opacity_vector[i];
		shs_backup_vector[i] = shs_vector[i];
	}
	// Finish the Re-Ordering

	max_scale_vector.resize(count);

	// Start BBX OCUPANCY
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(grid_gs_prefix_sum_cuda, 0, sizeof(int) * grid_count));

	vector<AABB> gs_aabbs(count);
	float* gs_aabbs_cuda;
	#pragma omp parallel for
	for (int gs_idx = 0; gs_idx < count; gs_idx++){
		Eigen::Quaternionf origin_q(rot_vector[gs_idx].rot[0], rot_vector[gs_idx].rot[1], rot_vector[gs_idx].rot[2], rot_vector[gs_idx].rot[3]);
		Eigen::Matrix3f rotation = origin_q.normalized().toRotationMatrix();
		gs_aabbs[gs_idx].xyz_max = pos_vector[gs_idx];
		gs_aabbs[gs_idx].xyz_min = pos_vector[gs_idx];

		Scale s_3d;
		if (opacity_vector[gs_idx] <= CUTOFF_ALPHA){
			s_3d.scale[0] = 0.0f; 
			s_3d.scale[1] = 0.0f; 
			s_3d.scale[2] = 0.0f;
			// std::cout << "gs_idx: " << gs_idx << " opacity: " << opacity_vector[gs_idx] << std::endl;
		}
		else{
			s_3d.scale[0] = sqrt(-2.0f*log(CUTOFF_ALPHA/opacity_vector[gs_idx])) * (scale_vector[gs_idx].scale[0]+0.0f);
			s_3d.scale[1] = sqrt(-2.0f*log(CUTOFF_ALPHA/opacity_vector[gs_idx])) * (scale_vector[gs_idx].scale[1]+0.0f);
			s_3d.scale[2] = sqrt(-2.0f*log(CUTOFF_ALPHA/opacity_vector[gs_idx])) * (scale_vector[gs_idx].scale[2]+0.0f);
		}

		scale_3d_clip[gs_idx] = s_3d;

		int max_variance_dim = 0;
		float scale_max = s_3d.scale[0];
		if (s_3d.scale[1] > scale_max){
			scale_max = s_3d.scale[1];
			max_variance_dim = 1;
		}
		if (s_3d.scale[2] > scale_max){
			scale_max = s_3d.scale[2];
			max_variance_dim = 2;
		}
		scale_3d_max[gs_idx] = scale_max;

		for (int dim = 0; dim < 3; dim++) {
			Pos sample_local_left(0.0f, 0.0f, 0.0f);
			sample_local_left(dim) = s_3d.scale[dim]; // Edit;
			sample_local_left = rotation * sample_local_left;
			Pos sample_local_right = -sample_local_left;
			sample_local_left = (pos_vector[gs_idx] + sample_local_left);
			sample_local_right = (pos_vector[gs_idx] + sample_local_right);

			gs_aabbs[gs_idx].xyz_min.x() = min(gs_aabbs[gs_idx].xyz_min.x(), sample_local_left.x());
			gs_aabbs[gs_idx].xyz_min.y() = min(gs_aabbs[gs_idx].xyz_min.y(), sample_local_left.y());
			gs_aabbs[gs_idx].xyz_min.z() = min(gs_aabbs[gs_idx].xyz_min.z(), sample_local_left.z());
			gs_aabbs[gs_idx].xyz_max.x() = max(gs_aabbs[gs_idx].xyz_max.x(), sample_local_left.x());
			gs_aabbs[gs_idx].xyz_max.y() = max(gs_aabbs[gs_idx].xyz_max.y(), sample_local_left.y());
			gs_aabbs[gs_idx].xyz_max.z() = max(gs_aabbs[gs_idx].xyz_max.z(), sample_local_left.z());

			gs_aabbs[gs_idx].xyz_min.x() = min(gs_aabbs[gs_idx].xyz_min.x(), sample_local_right.x());
			gs_aabbs[gs_idx].xyz_min.y() = min(gs_aabbs[gs_idx].xyz_min.y(), sample_local_right.y());
			gs_aabbs[gs_idx].xyz_min.z() = min(gs_aabbs[gs_idx].xyz_min.z(), sample_local_right.z());
			gs_aabbs[gs_idx].xyz_max.x() = max(gs_aabbs[gs_idx].xyz_max.x(), sample_local_right.x());
			gs_aabbs[gs_idx].xyz_max.y() = max(gs_aabbs[gs_idx].xyz_max.y(), sample_local_right.y());
			gs_aabbs[gs_idx].xyz_max.z() = max(gs_aabbs[gs_idx].xyz_max.z(), sample_local_right.z());
		}
	}

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&gs_aabbs_cuda, sizeof(AABB) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(gs_aabbs_cuda, gs_aabbs.data(), count*sizeof(AABB), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&scale_3d_max_cuda, sizeof(float) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_3d_max_cuda, scale_3d_max.data(), count*sizeof(float), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos_vector.data(), sizeof(Pos) * count, cudaMemcpyHostToDevice));


	GetBoxedGsNumPerGridCUDA(count, grid_step, grid_num, gs_aabbs_cuda, grid_gs_prefix_sum_cuda, min_xyz, padding);

	CUDA_SAFE_CALL_ALWAYS(cudaFree(gs_aabbs_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(scale_3d_max_cuda));


	grid_gs_prefix_sum_vector.resize(grid_count);
	std::cout << "grid_count: " << grid_count << std::endl;
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(grid_gs_prefix_sum_vector.data(), grid_gs_prefix_sum_cuda, grid_count*sizeof(int), cudaMemcpyDeviceToHost));

	valid_grid_num = 0;
	valid_grid_idx.clear();
	for (int i = 0; i < grid_count; i++){
		if (i == 0){
			if (grid_gs_prefix_sum_vector[i] != 0){
				valid_grid_num += 1;
				valid_grid_idx.push_back(i);
			}
		}
		else if ((grid_gs_prefix_sum_vector[i]-grid_gs_prefix_sum_vector[i-1]) != 0) {
			valid_grid_num += 1;
			valid_grid_idx.push_back(i);
		}
	}
	std::cout << "there are totally " << valid_grid_num << " grids containing gaussians boxes." << std::endl; 
	std::cout << "there are totally " << grid_gs_prefix_sum_vector[grid_count-1] << " grids--gaussian-boxes pairs." << std::endl; 

	CUDA_SAFE_CALL_ALWAYS(cudaFree(valid_grid_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&valid_grid_cuda, sizeof(int) * valid_grid_num));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(valid_grid_cuda, valid_grid_idx.data(), valid_grid_num*sizeof(int), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&grid_is_converged_cuda, sizeof(bool) * valid_grid_num));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(grid_is_converged_cuda, 0, sizeof(bool) * valid_grid_num));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&grid_nearly_converged_cuda, sizeof(bool) * valid_grid_num));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(grid_nearly_converged_cuda, 0, sizeof(bool) * valid_grid_num));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&grid_loss_sums_cuda, sizeof(float) * valid_grid_num));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(grid_loss_sums_cuda, 0.0, sizeof(float)* valid_grid_num));




	grided_gs_idx.resize(grid_gs_prefix_sum_vector[grid_count-1]);
	grid_used_gs.assign(grid_count, 0);

	// #pragma omp parallel for
	for (int i = 0; i < count; i++){
		int x_idx_min = max(int(floor((gs_aabbs[i].xyz_min.x() - min_xyz.x)/grid_step) - padding), 0);
		int y_idx_min = max(int(floor((gs_aabbs[i].xyz_min.y() - min_xyz.y)/grid_step) - padding), 0);
		int z_idx_min = max(int(floor((gs_aabbs[i].xyz_min.z() - min_xyz.z)/grid_step) - padding), 0);
		
		int x_idx_max = min(int(floor((gs_aabbs[i].xyz_max.x() - min_xyz.x)/grid_step) + padding), grid_num-1);
		int y_idx_max = min(int(floor((gs_aabbs[i].xyz_max.y() - min_xyz.y)/grid_step) + padding), grid_num-1);
		int z_idx_max = min(int(floor((gs_aabbs[i].xyz_max.z() - min_xyz.z)/grid_step) + padding), grid_num-1);

		for (int x_idx = x_idx_min; x_idx < x_idx_max + 1; x_idx++){
			for (int y_idx = y_idx_min; y_idx < y_idx_max + 1; y_idx++){
				for (int z_idx = z_idx_min; z_idx < z_idx_max + 1; z_idx++){
					int grid_idx = x_idx*grid_num*grid_num + y_idx*grid_num + z_idx;
					if (grid_idx == 0){
						grided_gs_idx[grid_used_gs[grid_idx]] = i;
					}
					else {
						grided_gs_idx[grid_gs_prefix_sum_vector[grid_idx-1] + grid_used_gs[grid_idx]] = i;
					}
					grid_used_gs[grid_idx] += 1;
				}
			}
		}
	}

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&grided_gs_idx_cuda, sizeof(int) * grid_gs_prefix_sum_vector[grid_count-1]));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(grided_gs_idx_cuda, grided_gs_idx.data(), grid_gs_prefix_sum_vector[grid_count-1]*sizeof(int), cudaMemcpyHostToDevice));
	// Finish BBX OCUPANCY
	auto end_bbx = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsedSeconds_bbx = end_bbx - start_bbx;
	std::cout << "BBX OCUPANCY took " << elapsedSeconds_bbx.count() << " seconds." << std::endl;


	
	samples_per_grid_dim = SAMPLES_PER_GRID;
	float samples_interval = grid_step / samples_per_grid_dim;

	std::vector<SamplePoint> grid_samples;
	for (int i = 0; i < valid_grid_num; i++){
		int grid_idx = valid_grid_idx[i];
		int x_idx = grid_idx / pow(grid_num, 2);
		int y_idx = (grid_idx - x_idx * pow(grid_num, 2)) / grid_num;
		int z_idx = grid_idx % grid_num;
		float grid_x_min = aabb_overall.xyz_min.x() + x_idx*grid_step + 0.5*samples_interval;
		float grid_y_min = aabb_overall.xyz_min.y() + y_idx*grid_step + 0.5*samples_interval;
		float grid_z_min = aabb_overall.xyz_min.z() + z_idx*grid_step + 0.5*samples_interval;
		for (int sample_idx = 0; sample_idx < pow(samples_per_grid_dim, 3); sample_idx++){
			int x_sample_idx = sample_idx / pow(samples_per_grid_dim, 2);
			int y_sample_idx = (sample_idx - x_sample_idx * pow(samples_per_grid_dim, 2)) / samples_per_grid_dim;
			int z_sample_idx = sample_idx % samples_per_grid_dim;
			Pos sample(grid_x_min+ x_sample_idx*samples_interval, grid_y_min+ y_sample_idx*samples_interval, grid_z_min+ z_sample_idx*samples_interval);
			sample_positions.push_back(sample);
			backup_sample_positions.push_back(sample);
			SamplePoint sp;
			grid_samples.push_back(sp);
		}
	}
	std::cout << "there are totally " << grid_samples.size() << " grid-samples." << std::endl; 

	// get initial grid idx for gaussians
	gs_init_grid_idx.resize(count);
	for (int i = 0; i < count; i++){
		int x_idx = int(floor((pos_vector[i].x() - min_xyz.x)/grid_step));
		int y_idx = int(floor((pos_vector[i].y() - min_xyz.y)/grid_step));
		int z_idx = int(floor((pos_vector[i].z() - min_xyz.z)/grid_step));
		int grid_idx = x_idx*grid_num*grid_num + y_idx*grid_num + z_idx;
		gs_init_grid_idx[i] = grid_idx;
	}
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&gs_init_grid_idx_cuda, sizeof(int) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(gs_init_grid_idx_cuda, gs_init_grid_idx.data(), count*sizeof(int), cudaMemcpyHostToDevice));
	
	return grid_samples;
}


void sibr::GaussianView::UpdateFeatures(){
	CopyFeatureFromCPU2GPU();
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(grid_is_converged_cuda, 0, sizeof(bool) * valid_grid_num));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(grid_nearly_converged_cuda, 0, sizeof(bool) * valid_grid_num));

	UpdateContainingRelationship();

	CudaRasterizer::Rasterizer::forward3d_grid(valid_grid_num, 3, 16, 
										count, samples_count_per_grid,
										valid_grid_cuda,
										grid_gs_prefix_sum_cuda,
										sample_pos_cuda,
										pos_cuda,
										rot_cuda,
										scale_orig_cuda,
										opacity_orig_cuda,
										shs_cuda,
										half_length_cuda,
										sigma_cuda,
										sigma_damp_cuda,
										cur_feature_cuda,
										cur_opacity_cuda,
										grided_gs_idx_cuda,
										grid_is_converged_cuda,
										grid_nearly_converged_cuda,
										opt_options_cuda,
										low_pass_filter_param,
										ada_lpf_ratio,
										min_xyz,
										grid_step,
										grid_num,
										gs_init_grid_idx_cuda,
										empty_grid_cuda,
										current_static_grids_cuda,
										high_quality);
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(total_feature_loss, 0.0, sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(total_shape_loss, 0.0, sizeof(float)));

	CudaRasterizer::Rasterizer::L1loss3d(valid_grid_num, 3, 16, 
										count, samples_count_per_grid,
										aim_feature_cuda,
										cur_feature_cuda,
										aim_opacity_cuda,
										cur_opacity_cuda,
										opacity_grad_cuda,
										feature_grad_cuda,
										total_feature_loss,
										total_shape_loss,
										grid_is_converged_cuda,
										grid_nearly_converged_cuda,
										grid_loss_sums_cuda,
										opt_options_cuda,
										empty_grid_cuda,
										adjust_op_range,
										high_quality);
	cudaDeviceSynchronize();
	
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(sample_feature_shs.data(), cur_feature_cuda, sizeof(SHs<3>) * sps.sample_points.size(), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(feature_opacity_vector.data(), cur_opacity_cuda, sizeof(float) * sps.sample_points.size(), cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(&loss, total_feature_loss, sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(&shape_loss, total_shape_loss, sizeof(float), cudaMemcpyDeviceToHost));
}

void sibr::GaussianView::GPUSetupSamplesFeatures(){
	CopyFeatureFromCPU2GPU();
	cudaDeviceSynchronize();
	UpdateContainingRelationship();
	auto start = std::chrono::steady_clock::now();
	CudaRasterizer::Rasterizer::forward3d_grid(valid_grid_num, 3, 16, 
										count, samples_count_per_grid,
										valid_grid_cuda,
										grid_gs_prefix_sum_cuda,
										sample_pos_cuda,
										pos_cuda,
										rot_cuda,
										scale_orig_cuda,
										opacity_orig_cuda,
										shs_cuda,
										half_length_cuda,
										sigma_cuda,
										sigma_damp_cuda,
										aim_feature_cuda,
										aim_opacity_cuda,
										grided_gs_idx_cuda,
										grid_is_converged_cuda,
										grid_nearly_converged_cuda,
										opt_options_cuda,
										low_pass_filter_param,
										ada_lpf_ratio,
										min_xyz,
										grid_step,
										grid_num,
										gs_init_grid_idx_cuda,
										empty_grid_cuda,
										current_static_grids_cuda,
										high_quality);


	cudaDeviceSynchronize();
	auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedSeconds = end - start;
    std::cout << "forward 3d took " << elapsedSeconds.count() << " seconds." << std::endl;

	aim_feature_shs.resize(sps.sample_points.size());
	sample_feature_shs.resize(sps.sample_points.size());
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(aim_feature_shs.data(), aim_feature_cuda, sizeof(SHs<3>) * sps.sample_points.size(), cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(sample_feature_shs.data(), aim_feature_cuda, sizeof(SHs<3>) * sps.sample_points.size(), cudaMemcpyDeviceToHost));

	aim_opacity.resize(sps.sample_points.size());
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(aim_opacity.data(), aim_opacity_cuda, sizeof(float) * sps.sample_points.size(), cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(feature_opacity_vector.data(), aim_opacity_cuda, sizeof(float) * sps.sample_points.size(), cudaMemcpyDeviceToHost));

	JudgeEmptyGrid();

}

void sibr::GaussianView::JudgeEmptyGrid(){
	empty_grid.assign(valid_grid_num, 0);

	std::vector<bool> bad_grid(grid_count, true);

	int samples_per_grid = pow(samples_per_grid_dim, 3);

	// Padding first round
	for (int i = 0; i < valid_grid_num; i++){
		float total_opacity = 0.0f;

		for (int j = 0; j < samples_per_grid; j++){
			total_opacity += aim_opacity[i*samples_per_grid + j];
		}

		if (total_opacity > 1e-6){
			int idx = valid_grid_idx[i];
			int z_idx = idx % grid_num;
			int y_idx = (idx / grid_num) % grid_num;
			int x_idx = idx / (grid_num * grid_num);

			bad_grid[idx] = false;

			int left_idx = max(x_idx - 1, 0)*(grid_num * grid_num) + y_idx*grid_num + z_idx;
			bad_grid[left_idx] = false;
			int right_idx = min(x_idx + 1, grid_num-1)*(grid_num * grid_num) + y_idx*grid_num + z_idx;
			bad_grid[right_idx] = false;
			int down_idx = x_idx*(grid_num * grid_num) + max(y_idx - 1, 0)*grid_num + z_idx;
			bad_grid[down_idx] = false;
			int up_idx = x_idx*(grid_num * grid_num) + min(y_idx + 1, grid_num-1)*grid_num + z_idx;
			bad_grid[up_idx] = false;
			int back_idx = x_idx*(grid_num * grid_num) + y_idx*grid_num + max(z_idx - 1, 0);
			bad_grid[back_idx] = false;
			int front_idx = x_idx*(grid_num * grid_num) + y_idx*grid_num + min(z_idx + 1, grid_num-1);
			bad_grid[front_idx] = false;
		}
	}
	for (int i = 0; i < valid_grid_num; i++){
		int idx = valid_grid_idx[i];
		if (bad_grid[idx] == true){
			empty_grid[i] = 1;
		}
	}

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(empty_grid_cuda, empty_grid.data(), sizeof(int) * empty_grid.size(), cudaMemcpyHostToDevice));

}

void sibr::GaussianView::ExpandOpRange(){
	bool converged_grid[valid_grid_num] = {true};
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(converged_grid, grid_is_converged_cuda, sizeof(bool) * valid_grid_num, cudaMemcpyDeviceToHost));

	std::vector<bool> bad_grid(grid_count, true);

	int samples_per_grid = pow(samples_per_grid_dim, 3);

	// Expand first round
	for (int i = 0; i < valid_grid_num; i++){
		if (!converged_grid[i]){
			int idx = valid_grid_idx[i];
			int z_idx = idx % grid_num;
			int y_idx = (idx / grid_num) % grid_num;
			int x_idx = idx / (grid_num * grid_num);

			bad_grid[idx] = false;

			int left_idx = max(x_idx - 1, 0)*(grid_num * grid_num) + y_idx*grid_num + z_idx;
			bad_grid[left_idx] = false;
			int right_idx = min(x_idx + 1, grid_num-1)*(grid_num * grid_num) + y_idx*grid_num + z_idx;
			bad_grid[right_idx] = false;
			int down_idx = x_idx*(grid_num * grid_num) + max(y_idx - 1, 0)*grid_num + z_idx;
			bad_grid[down_idx] = false;
			int up_idx = x_idx*(grid_num * grid_num) + min(y_idx + 1, grid_num-1)*grid_num + z_idx;
			bad_grid[up_idx] = false;
			int back_idx = x_idx*(grid_num * grid_num) + y_idx*grid_num + max(z_idx - 1, 0);
			bad_grid[back_idx] = false;
			int front_idx = x_idx*(grid_num * grid_num) + y_idx*grid_num + min(z_idx + 1, grid_num-1);
			bad_grid[front_idx] = false;
		}
	}
	for (int i = 0; i < valid_grid_num; i++){
		int idx = valid_grid_idx[i];
		if (bad_grid[idx] == false){
			converged_grid[i] = false;
		}
	}

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(grid_is_converged_cuda, converged_grid, sizeof(bool) * valid_grid_num, cudaMemcpyHostToDevice));
}


void sibr::GaussianView::GPUOptimize(){
	auto start = std::chrono::steady_clock::now();
	cudaDeviceSynchronize();
	CudaRasterizer::Rasterizer::forward3d_grid(valid_grid_num, 3, 16, 
										count, samples_count_per_grid,
										valid_grid_cuda,
										grid_gs_prefix_sum_cuda,
										sample_pos_cuda,
										pos_cuda,
										rot_cuda,
										scale_orig_cuda,
										opacity_orig_cuda,
										shs_cuda,
										half_length_cuda,
										sigma_cuda,
										sigma_damp_cuda,
										cur_feature_cuda,
										cur_opacity_cuda,
										grided_gs_idx_cuda,
										grid_is_converged_cuda,
										grid_nearly_converged_cuda,
										opt_options_cuda,
										low_pass_filter_param,
										ada_lpf_ratio,
										min_xyz,
										grid_step,
										grid_num,
										gs_init_grid_idx_cuda,
										empty_grid_cuda,
										current_static_grids_cuda,
										high_quality);

	cudaDeviceSynchronize();
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(total_feature_loss, 0.0, sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(total_shape_loss, 0.0, sizeof(float)));

	CudaRasterizer::Rasterizer::L1loss3d(valid_grid_num, 3, 16, 
										count, samples_count_per_grid,
										aim_feature_cuda,
										cur_feature_cuda,
										aim_opacity_cuda,
										cur_opacity_cuda,
										opacity_grad_cuda,
										feature_grad_cuda,
										total_feature_loss,
										total_shape_loss,
										grid_is_converged_cuda,
										grid_nearly_converged_cuda,
										grid_loss_sums_cuda,
										opt_options_cuda,
										empty_grid_cuda,
										adjust_op_range,
										high_quality);

	
	cudaDeviceSynchronize();
	CudaRasterizer::Rasterizer::backward3d_grid(valid_grid_num, 3, 16, 
										count, samples_count_per_grid,
										valid_grid_cuda,
										grid_gs_prefix_sum_cuda,
										sample_pos_cuda,
										pos_cuda,
										rot_cuda,
										scale_orig_cuda,
										opacity_orig_cuda,
										shs_cuda,
										half_length_cuda,
										sigma_cuda,
										sigma_damp_cuda,
										opacity_grad_cuda,
										feature_grad_cuda,
										dF_dopacity_cuda,
										dF_dshs_cuda,
										dF_dpos_cuda,
										dF_drot_cuda,
										dF_dscale_cuda,
										dF_dcov3D,
										grided_gs_idx_cuda,
										grid_is_converged_cuda,
										opt_options_cuda,
										min_xyz,
										grid_step,
										grid_num,
										ada_lpf_ratio,
										empty_grid_cuda,
										current_static_grids_cuda,
										moved_gaussians_cuda,
										high_quality);
	cudaDeviceSynchronize();
	CudaRasterizer::Rasterizer::update3d(count, 3, 16,
										dF_dopacity_cuda,
										dF_dshs_cuda,
										dF_dpos_cuda,
										dF_drot_cuda,
										dF_dscale_cuda,
										opacity_orig_cuda,
										shs_cuda,
										pos_cuda,
										rot_cuda,
										scale_orig_cuda,
										m_opacity_cuda,
										v_opacity_cuda,
										m_shs_cuda,
										v_shs_cuda,
										m_pos_cuda,
										v_pos_cuda,
										m_rot_cuda,
										v_rot_cuda,
										m_scale_cuda,
										v_scale_cuda,
										max_scale_cuda,
										step,
										opt_options_cuda,
										learning_rate_cuda,
										_optimize_steps,
										moved_gaussians_cuda);
	cudaDeviceSynchronize();
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(&cur_step, step, sizeof(int), cudaMemcpyDeviceToHost));
	std::cout << "step: " << cur_step << std::endl;

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedSeconds = end - start;
	std::cout << "GPUOptimize took " << elapsedSeconds.count() << " seconds." << std::endl;
}


void sibr::GaussianView::CopyFeatureFromCPU2GPU(){
	auto start_copy = std::chrono::steady_clock::now();

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_cuda, pos_vector.data(), sizeof(Pos) * count, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_cuda, rot_vector.data(), sizeof(Rot) * count, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_cuda, shs_vector.data(), sizeof(SHs<3>) * count, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(sample_pos_cuda, sample_positions.data(), sizeof(Pos) * sample_positions.size(), cudaMemcpyHostToDevice));

	opacity_orig_vector.resize(count);
	scale_orig_vector.resize(count);
	#pragma omp parallel for
	for (int i = 0; i < count; i++) {
		opacity_orig_vector[i] = inverse_sigmoid(opacity_vector[i]);
		scale_orig_vector[i].scale[0] = log(scale_vector[i].scale[0]);
		scale_orig_vector[i].scale[1] = log(scale_vector[i].scale[1]);
		scale_orig_vector[i].scale[2] = log(scale_vector[i].scale[2]);
	}
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_orig_cuda, opacity_orig_vector.data(), sizeof(float) * count, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_orig_cuda, scale_orig_vector.data(), sizeof(Scale) * count, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opt_options_cuda, _optimize_options, sizeof(bool) * 12, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(learning_rate_cuda, _learning_rate, sizeof(float) * 5, cudaMemcpyHostToDevice));

	auto end_copy = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsedSeconds_copy = end_copy - start_copy;
	std::cout << "CopyFeatureFromCPU2GPU took " << elapsedSeconds_copy.count() << " seconds." << std::endl;
}

void sibr::GaussianView::CopyPartialInfoFromGPU2CPU(){ 
	// CopyPartialInfoFromGPU2CPU for UpdateContainingRelationship
	auto start_copy = std::chrono::steady_clock::now();

	cudaDeviceSynchronize();
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_vector.data(), pos_cuda, sizeof(Pos) * count, cudaMemcpyDeviceToHost)); 
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_vector.data(), rot_cuda, sizeof(Rot) * count, cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_orig_vector.data(), opacity_orig_cuda, sizeof(float) * count, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_orig_vector.data(), scale_orig_cuda, sizeof(Scale) * count, cudaMemcpyDeviceToHost));

	#pragma omp parallel for
	for (int i = 0; i < count; i++) {
		opacity_vector[i] = sigmoid(opacity_orig_vector[i]);
		scale_vector[i].scale[0] = exp(scale_orig_vector[i].scale[0]);
		scale_vector[i].scale[1] = exp(scale_orig_vector[i].scale[1]);
		scale_vector[i].scale[2] = exp(scale_orig_vector[i].scale[2]);
		float norm = 0.0f;
		for (int j = 0; j < 4; j++) {
			norm += rot_vector[i].rot[j] * rot_vector[i].rot[j];
		}
		norm = sqrt(norm);
		for (int j = 0; j < 4; j++) {
			rot_vector[i].rot[j] = rot_vector[i].rot[j] / norm;
		}
	}

	auto end_copy = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsedSeconds_copy = end_copy - start_copy;
	std::cout << "CopyPartialInfoFromGPU2CPU took " << elapsedSeconds_copy.count() << " seconds." << std::endl;
}

void sibr::GaussianView::CopyFeatureFromGPU2CPU(){
	auto start_copy = std::chrono::steady_clock::now();

	cudaDeviceSynchronize();
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(pos_vector.data(), pos_cuda, sizeof(Pos) * count, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(shs_vector.data(), shs_cuda, sizeof(SHs<3>) * count, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(rot_vector.data(), rot_cuda, sizeof(Rot) * count, cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(sample_feature_shs.data(), cur_feature_cuda, sizeof(SHs<3>) * sps.sample_points.size(), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(feature_opacity_vector.data(), cur_opacity_cuda, sizeof(float) * sps.sample_points.size(), cudaMemcpyDeviceToHost));

	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_orig_vector.data(), opacity_orig_cuda, sizeof(float) * count, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_orig_vector.data(), scale_orig_cuda, sizeof(Scale) * count, cudaMemcpyDeviceToHost));

	#pragma omp parallel for
	for (int i = 0; i < count; i++) {
		opacity_vector[i] = sigmoid(opacity_orig_vector[i]);
		scale_vector[i].scale[0] = exp(scale_orig_vector[i].scale[0]);
		scale_vector[i].scale[1] = exp(scale_orig_vector[i].scale[1]);
		scale_vector[i].scale[2] = exp(scale_orig_vector[i].scale[2]);

		for (int j = 0; j < 3; j++) {
			if (std::isinf(scale_vector[i].scale[j]) || std::isnan(scale_vector[i].scale[j])){
				scale_vector[i].scale[i] = 1e-8f;
			}
		}

		float norm = 0.0f;
		for (int j = 0; j < 4; j++) {
			norm += rot_vector[i].rot[j] * rot_vector[i].rot[j];
		}
		norm = sqrt(norm);
		for (int j = 0; j < 4; j++) {
			rot_vector[i].rot[j] = rot_vector[i].rot[j] / norm;
		}
	}

	auto end_copy = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsedSeconds_copy = end_copy - start_copy;
	std::cout << "CopyFeatureFromGPU2CPU took " << elapsedSeconds_copy.count() << " seconds." << std::endl;
}

void sibr::GaussianView::UpdateGsCoverRange(){
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(opacity_orig_vector.data(), opacity_orig_cuda, sizeof(float) * count, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(scale_orig_vector.data(), scale_orig_cuda, sizeof(Scale) * count, cudaMemcpyDeviceToHost));

	#pragma omp parallel for
	for (int i = 0; i < count; i++) {
		opacity_vector[i] = sigmoid(opacity_orig_vector[i]);
		scale_vector[i].scale[0] = exp(scale_orig_vector[i].scale[0]);
		scale_vector[i].scale[1] = exp(scale_orig_vector[i].scale[1]);
		scale_vector[i].scale[2] = exp(scale_orig_vector[i].scale[2]);
	}
}

void sibr::GaussianView::RefreshAdam(){
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(m_shs_cuda, 0.0, sizeof(SHs<3>) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(v_shs_cuda, 0.0, sizeof(SHs<3>) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(m_opacity_cuda, 0.0, sizeof(float) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(v_opacity_cuda, 0.0, sizeof(float) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(m_pos_cuda, 0.0, sizeof(Pos) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(v_pos_cuda, 0.0, sizeof(Pos) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(m_rot_cuda, 0.0, sizeof(Rot) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(v_rot_cuda, 0.0, sizeof(Rot) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(m_scale_cuda, 0.0, sizeof(Scale) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(v_scale_cuda, 0.0, sizeof(Scale) * count));
	CUDA_SAFE_CALL_ALWAYS(cudaMemset(step, 0, sizeof(int)));
	cur_step = 0;
}


void sibr::GaussianView::SetupUniformSamples(std::vector<SamplePoint>& current_sps){
	getOverallAABB();
	num_samples_per_dim = NUM_SAMPLES_PER_DIM;
	int uniform_samples = pow(num_samples_per_dim, 3);

	x_step = (aabb_overall.xyz_max.x() - aabb_overall.xyz_min.x()) / num_samples_per_dim;
	y_step = (aabb_overall.xyz_max.y() - aabb_overall.xyz_min.y()) / num_samples_per_dim;
	z_step = (aabb_overall.xyz_max.z() - aabb_overall.xyz_min.z()) / num_samples_per_dim;

	#pragma omp parallel for
	for (int i = 0; i < uniform_samples; i++){
		int x_idx = i / pow(num_samples_per_dim, 2);
		int y_idx = (i - x_idx * pow(num_samples_per_dim, 2)) / num_samples_per_dim;
		int z_idx = i % num_samples_per_dim;
		Pos sample(aabb_overall.xyz_min.x()+ x_idx*x_step, aabb_overall.xyz_min.y()+ y_idx*y_step, aabb_overall.xyz_min.z()+ z_idx*z_step);
		sample_positions_new[i] = sample;
		// SPACE FOR BACKUP
		SamplePoint sp;
		current_sps[i] = sp;
	}
	// SPACE FOR INITIALIZE SPS
}

void sibr::GaussianView::GetEndPoints(){
	ends_vector.resize(count);
	ends_vector_backup.resize(count);
	#pragma omp parallel for
	for (int gs_idx = 0; gs_idx < count; gs_idx++){
		std::array<int, 3> indices = OrderIndices(scale_vector[gs_idx]);
		EndPoint endpoint;
		endpoint.axises[0] = indices[2];
		endpoint.axises[1] = indices[1];
		endpoint.axises[2] = indices[0];

		Eigen::Quaternionf origin_q(rot_vector[gs_idx].rot[0], rot_vector[gs_idx].rot[1], rot_vector[gs_idx].rot[2], rot_vector[gs_idx].rot[3]);
		Eigen::Matrix3f rotation = origin_q.normalized().toRotationMatrix();

		for (int i = 0; i < 3; i++){
			Pos endpos(0.0f, 0.0f, 0.0f);
			endpos(i) = (scale_vector[gs_idx].scale[i] + axis_padding)*end_coeff;
			endpos = rotation * endpos;
			endpoint.ends[i].first = pos_vector[gs_idx] + endpos;
			endpoint.ends[i].second = pos_vector[gs_idx] - endpos;
		}
		ends_vector[gs_idx] = endpoint;
	}
	ends_vector_backup = ends_vector;
	return;
}

void sibr::GaussianView::GetAdaLpfRatio(){
	int samples_per_grid = pow(samples_per_grid_dim, 3);

	#pragma omp parallel for
	for (int i = 0; i < valid_grid_num; i++){
		int grid_idx = valid_grid_idx[i];

		Eigen::Matrix<float, 3, 8> Q;
		Eigen::Matrix<float, 3, 8> P;
		// Load Matrix Q and P
		//    15 - - 63
		//   /|     /|			y z
		// 12 - - 60 |			|/
		// |  03 -|- 51			--x
		// | /    | /
		// 00 - - 48
		// init, left(z), top(y), forward(x) -> 0, 3, 12, 48
		int base = i*samples_per_grid;
		int x_on = (samples_per_grid_dim - 1)*(pow(samples_per_grid_dim, 2));
		int y_on = (samples_per_grid_dim - 1)*samples_per_grid_dim;
		int z_on = (samples_per_grid_dim - 1);
		Pos base_pose = sample_positions[base];
		float base_s = (float)(samples_per_grid_dim - 1);

		Eigen::Vector3f v3;
		for (int col_idx = 0; col_idx < 8; col_idx++){
			int idx = base;
			v3 << 0.0f, 0.0f, 0.0f;
			if (col_idx % 2 == 1){ // x_on
				v3(0) = 1.0f; idx += x_on; 
			}
			if ((col_idx / 2) % 2 == 1){ // y_on
				v3(1) = 1.0f; idx += y_on; 
			}
			if ((col_idx / 4) % 2 == 1){ // z_on
				v3(2) = 1.0f; idx += z_on; 
			}
			P.col(col_idx) = (sample_positions[idx] - base_pose) / base_s;
			Q.col(col_idx) = v3;
		}

		Eigen::Matrix<float, 24, 9> A;
		Eigen::Matrix<float, 24, 1> b;
		A.setZero();
		b.setZero();

		int idx = 0;
		for (int i = 0; i < 8; i++){
			for (int j = 0 ; j < 3; j++){
				for (int k = 0; k < 3; k++){
					A(idx, j*3 + k) = Q(k, i);
				}
				b(idx) = P(j, i);
				idx++;
			}
		}

		Eigen::MatrixXf AtA = A.transpose()*A;
		Eigen::Matrix<float, 9, 1> Atb = A.transpose() * b;
		Eigen::VectorXf x = (AtA).ldlt().solve(Atb);
		Eigen::MatrixXf M(3, 3);
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j){
				M(i, j) = x(i*3 + j);
			}
		}

		ada_lpf_vector[grid_idx] = M * M.transpose() * lpf_parameter;

		// if (i == 0) {
		// 	std::cout<< "ada_lpf_vector: \n" << ada_lpf_vector[grid_idx] << std::endl;
		// 	std::cout<< "M: \n" << M << std::endl;
		// }
	}


	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(ada_lpf_ratio, ada_lpf_vector.data(), sizeof(Eigen::Matrix3f)*grid_count, cudaMemcpyHostToDevice));

	std::cout<< "GetAdaLpfRatio " << std::endl;

	return;
}


void sibr::GaussianView::TakeSnapshot(){
	Snapshot s;
	s.gs_num = count;
 	s.pos_vector = pos_vector;
	s.rot_vector = rot_vector;
	s.opacity_vector = opacity_vector;
	s.shs_vector = shs_vector;
	s.scale_vector = scale_vector;
	s.vertices = vertices;

	s.scale_3d_clip = scale_3d_clip;
	s.scale_3d_max = scale_3d_max;
	
	s.sp_num = sample_positions.size();
	// s.samples_pos_payload_vector = samples_pos_payload_vector;
	s.sample_positions = sample_positions;
	s.sps = sps;

	s.deform_graph = deform_graph;
	s.aim_centers = aim_centers;
	s.aim_centers_radius = aim_centers_radius;
	s.static_indices = static_indices;
	s.control_indices = control_indices;
	s.indices_blocks = indices_blocks;
	s.block_num = block_num;
	s.blocks_type = blocks_type;

	snapshots.push_back(s);
}

void sibr::GaussianView::LoadSnapshot(int idx){
	if (idx >= snapshots.size()) return;
	Snapshot s = snapshots[idx];
	pos_vector = s.pos_vector;
	rot_vector = s.rot_vector;
	opacity_vector = s.opacity_vector;
	shs_vector = s.shs_vector;
	scale_vector = s.scale_vector;
	vertices = s.vertices;

	scale_3d_clip = s.scale_3d_clip;
	scale_3d_max = s.scale_3d_max;
	
	// samples_pos_payload_vector = s.samples_pos_payload_vector;
	sample_positions = s.sample_positions;
	sps = s.sps;

	deform_graph = s.deform_graph;
	aim_centers = s.aim_centers;
	aim_centers_radius = s.aim_centers_radius;
	static_indices = s.static_indices;
	control_indices = s.control_indices;
	indices_blocks = s.indices_blocks;
	block_num = s.block_num;
	blocks_type = s.blocks_type;
	
	CUDA_SAFE_CALL_ALWAYS(cudaFree(sample_pos_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(cur_feature_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(feature_grad_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaFree(cur_opacity_cuda));

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&sample_pos_cuda, sizeof(Pos) * sps.sample_points.size()));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cur_feature_cuda, sizeof(SHs<3>) * sps.sample_points.size()));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&feature_grad_cuda, sizeof(SHs<3>) * sps.sample_points.size()));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cur_opacity_cuda, sizeof(float) * sps.sample_points.size()));

	UpdateIndicies(deform_graph);
	UpdateFeatures();
}


void sibr::GaussianView::RebuildGraph(){
	ResetAll();
	static_indices.clear();
	control_indices.clear();
	indices_blocks.clear();
	aim_centers.clear();
	aim_centers_radius.clear();
	block_num = 0;

	std::vector<int> node_index;
	node_index = farthest_control_points_sampling(cand_points, node_num, _surface_graph);

	std::vector<Node> nodes;
    for (int i : node_index)
    {
        Node node;
        node.Position = cand_points[i];
        node.Vertex_index = i;
        nodes.push_back(node);
    }
	deform_graph = DeformGraph(nodes, knn_k);
	setupWeights(deform_graph);
	setupWeightsforSamples(deform_graph);
	setupWeightsforMesh(deform_graph);
	setupWeightsforSoup(deform_graph);
	setupWeightsforEnds(deform_graph);

	ReloadAimPositions();

	CUDA_SAFE_CALL_ALWAYS(cudaFree(q_vector_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&q_vector_cuda, sizeof(Eigen::Quaternionf) * nodes.size()));
}


void sibr::GaussianView::RecordDeformation(){
	// get the filename to store
	std::string output_path(pcd_filepath);
	output_path.replace(pcd_filepath.length() - 4, 4, std::string(_deform_filepath));
	std::ofstream outfile(output_path);

	// the node type(on Mesh or Gaussians)
	if (nodes_on_mesh){
		outfile << 1 << " ";
	}
	else {
		outfile << 0 << " ";
	}
	// record the gaussians/node indices
	outfile << "Nodes: ";
	outfile << deform_graph.nodes.size() << " ";
	for (unsigned int i = 0; i < deform_graph.nodes.size(); i++) {
		outfile << deform_graph.nodes[i].Vertex_index << " ";
	}

	// record the operations num
	outfile << "Total_Operations: " << history.total_operations << " ";
	outfile << "Move_Operations: " << history.move_operations << " ";

	// record the operation types
	outfile << "Operation_Types: " << history.operation_types.size() << " ";
	for (unsigned int i = 0; i < history.operation_types.size(); i++) {
		outfile << history.operation_types[i] << " ";
	}

	// record the block nodes
	outfile << "Block_Nodes: " << history.block_nodes.size() << " ";
	for (unsigned int i = 0; i < history.block_nodes.size(); i++) {
		outfile << history.block_nodes[i].size() << " ";
		for (unsigned int j = 0; j < history.block_nodes[i].size(); j++) {
			outfile << history.block_nodes[i][j] << " ";
		}
	}

	// record the mouse movements
	outfile << "Mouse_Movements: " << history.mouse_movements.size() << " ";
	for (unsigned int i = 0; i < history.mouse_movements.size(); i++) {
		outfile << history.mouse_movements[i].size() << " ";
		for (unsigned int j = 0; j < history.mouse_movements[i].size(); j++) {
			outfile << history.mouse_movements[i][j](0) << " ";
			outfile << history.mouse_movements[i][j](1) << " ";
			outfile << history.mouse_movements[i][j](2) << " ";
		}
	}

	// record the blocks types during deformations
	outfile << "Blocks_Types_Moves: " << history.blocks_types_moves.size() << " ";
	for (unsigned int i = 0; i < history.blocks_types_moves.size(); i++) {
		outfile << history.blocks_types_moves[i].size() << " ";
		for (unsigned int j = 0; j < history.blocks_types_moves[i].size(); j++) {
			outfile << history.blocks_types_moves[i][j] << " ";
		}
	}

	// record whether the energy is on center 
	outfile << "Energy_on_Center: " << history.energy_on_centers.size() << " ";
	for (unsigned int i = 0; i < history.energy_on_centers.size(); i++) {
		outfile << history.energy_on_centers[i] << " ";
	}

	// record the twist axis
	outfile << "Twist_Axis: " << history.twist_axis.size() << " ";
	for (unsigned int i = 0; i < history.twist_axis.size(); i++) {
		outfile << history.twist_axis[i](0) << " ";
		outfile << history.twist_axis[i](1) << " ";
		outfile << history.twist_axis[i](2) << " ";
		outfile << history.twist_axis[i](3) << " ";
	}

	outfile.close();
	return;
}



void sibr::GaussianView::LoadMeshForGraph(){
	simplified_points.clear();
	simplified_points_backup.clear();
	simplified_vertices.clear();

	std::string simplifiedfile(input_graph_path);
	LoadMeshPoints(simplifiedfile, simplified_points);
	LoadMeshPoints(simplifiedfile, simplified_points_backup);
	simplified_vertices.resize(simplified_points.size());
	std::cout << "Size of simplified_points: " << simplified_points.size() << std::endl;
	for (unsigned int i = 0; i < simplified_points.size(); i++){
		simplified_vertices[i].index = i;
	}

	if (nodes_on_mesh){
		node_num = simplified_points.size();
	}

	RebuildGraph();
	return;
}

void sibr::GaussianView::LoadDeformation(){
	std::ifstream infile(input_deform_path);

	// load the nodes type
	int temp_type;
	infile >> temp_type;
	if (temp_type == 1){
		nodes_on_mesh = true;
	}
	else {
		nodes_on_mesh = false;
	}

	ResetAll();

	// load the node indices
	std::string varName;
    infile >> varName;
	std::cout << "Load " << varName << std::endl;
	int size, size_inner;
	infile >> size;

	std::vector<int> node_indices;
	for (int i = 0; i < size; i++){
		int temp_int;
		infile >> temp_int;
		node_indices.push_back(temp_int);
	}
	// Re-set up the deform graph
	std::vector<Node> nodes;
	for (int i : node_indices){
		Node node;
		node.Position = cand_points[i];
		node.Vertex_index = i;
		nodes.push_back(node);
	}
	deform_graph = DeformGraph(nodes, knn_k);
	setupWeights(deform_graph);
	setupWeightsforSamples(deform_graph);
	setupWeightsforMesh(deform_graph);
	setupWeightsforSoup(deform_graph);
	setupWeightsforEnds(deform_graph);

	CUDA_SAFE_CALL_ALWAYS(cudaFree(q_vector_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&q_vector_cuda, sizeof(Eigen::Quaternionf) * nodes.size()));

	// clean the history
	history = DeformHistory();

	// load the operations num
	infile >> varName;
	infile >> history.total_operations;
	infile >> varName;
	infile >> history.move_operations;

	// load the operation types
	infile >> varName;
	cout << varName << endl;
	infile >> size;
	for (unsigned int i = 0; i < size; i++){
		int temp_int;
		infile >> temp_int;
		history.operation_types.push_back(temp_int);
	}

	// load the block nodes
	infile >> varName;
	cout << varName << endl;
	infile >> size;
	for (unsigned int i = 0; i < size; i++) {
		infile >> size_inner;
		history.block_nodes.push_back(vector<unsigned int>());
		for (unsigned int j = 0; j < size_inner; j++) {
			unsigned int temp_uint;
			infile >> temp_uint;
			history.block_nodes[i].push_back(temp_uint);
		}
	}

	// load the mouse movements
	infile >> varName;
	cout << varName << endl;
	infile >> size;
	for (unsigned int i = 0; i < size; i++) {
		infile >> size_inner;
		history.mouse_movements.push_back(vector<Pos>());
		for (unsigned int j = 0; j < size_inner; j++){
			Pos temp_pos;
			infile >> temp_pos(0);
			infile >> temp_pos(1);
			infile >> temp_pos(2);
			history.mouse_movements[i].push_back(temp_pos);
		}
	}

	// load the blocks types during deformations
	infile >> varName;
	cout << varName << endl;
	infile >> size;
	for (unsigned int i = 0; i < size; i++) {
		infile >> size_inner;
		history.blocks_types_moves.push_back(vector<int>());
		for (unsigned int j = 0; j < size_inner; j++) {
			int temp_int;
			infile >> temp_int;
			history.blocks_types_moves[i].push_back(temp_int);
		}
	}

	// load the operation types
	infile >> varName;
	cout << varName << endl;
	infile >> size;
	for (unsigned int i = 0; i < size; i++){
		int temp_int;
		infile >> temp_int;
		history.energy_on_centers.push_back(temp_int);
	}

	// load the twist axis
	infile >> varName;
	cout << varName << endl;
	infile >> size;
	for (unsigned int i = 0; i < size; i++) {
		Vector4f temp_vector;
		infile >> temp_vector(0);
		infile >> temp_vector(1);
		infile >> temp_vector(2);
		infile >> temp_vector(3);
		history.twist_axis.push_back(temp_vector);
	}

	infile.close();
	return;
}


void sibr::GaussianView::LoadDeformation_wo_rebuild(){
	std::ifstream infile(input_deform_path);

	// load the nodes type
	int temp_type;
	infile >> temp_type;

	ResetAll();

	// load the node indices
	std::string varName;
    infile >> varName;
	std::cout << "Load " << varName << std::endl;
	int size, size_inner;
	infile >> size;

	std::vector<int> node_indices;
	for (int i = 0; i < size; i++){
		int temp_int;
		infile >> temp_int;
		node_indices.push_back(temp_int);
	}
	// Re-set up the deform graph
	std::vector<Node> nodes;
	for (int i : node_indices){
		Node node;
		node.Position = cand_points[i];
		node.Vertex_index = i;
		nodes.push_back(node);
	}

	CUDA_SAFE_CALL_ALWAYS(cudaFree(q_vector_cuda));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&q_vector_cuda, sizeof(Eigen::Quaternionf) * nodes.size()));

	// clean the history
	history = DeformHistory();

	// load the operations num
	infile >> varName;
	infile >> history.total_operations;
	infile >> varName;
	infile >> history.move_operations;

	// load the operation types
	infile >> varName;
	cout << varName << endl;
	infile >> size;
	for (unsigned int i = 0; i < size; i++){
		int temp_int;
		infile >> temp_int;
		history.operation_types.push_back(temp_int);
	}

	// load the block nodes
	infile >> varName;
	cout << varName << endl;
	infile >> size;
	for (unsigned int i = 0; i < size; i++) {
		infile >> size_inner;
		history.block_nodes.push_back(vector<unsigned int>());
		for (unsigned int j = 0; j < size_inner; j++) {
			unsigned int temp_uint;
			infile >> temp_uint;
			history.block_nodes[i].push_back(temp_uint);
		}
	}

	// load the mouse movements
	infile >> varName;
	cout << varName << endl;
	infile >> size;
	for (unsigned int i = 0; i < size; i++) {
		infile >> size_inner;
		history.mouse_movements.push_back(vector<Pos>());
		for (unsigned int j = 0; j < size_inner; j++){
			Pos temp_pos;
			infile >> temp_pos(0);
			infile >> temp_pos(1);
			infile >> temp_pos(2);
			history.mouse_movements[i].push_back(temp_pos);
		}
	}

	// load the blocks types during deformations
	infile >> varName;
	cout << varName << endl;
	infile >> size;
	for (unsigned int i = 0; i < size; i++) {
		infile >> size_inner;
		history.blocks_types_moves.push_back(vector<int>());
		for (unsigned int j = 0; j < size_inner; j++) {
			int temp_int;
			infile >> temp_int;
			history.blocks_types_moves[i].push_back(temp_int);
		}
	}

	// load the operation types
	infile >> varName;
	cout << varName << endl;
	infile >> size;
	for (unsigned int i = 0; i < size; i++){
		int temp_int;
		infile >> temp_int;
		history.energy_on_centers.push_back(temp_int);
	}

	// load the twist axis
	infile >> varName;
	cout << varName << endl;
	infile >> size;
	for (unsigned int i = 0; i < size; i++) {
		Vector4f temp_vector;
		infile >> temp_vector(0);
		infile >> temp_vector(1);
		infile >> temp_vector(2);
		infile >> temp_vector(3);
		history.twist_axis.push_back(temp_vector);
	}

	infile.close();
	return;
}


void sibr::GaussianView::RunHistoricalDeform(){

	add_block_op_idx = 0;
	mouse_move_op_idx = 0;
	run_history = true;
	history_step = 0;

	return;
};

void sibr::GaussianView::CleanDeformHistory(){
	history = DeformHistory();
	return;
};



void sibr::GaussianView::ReloadAimPositions(){
	aim_positions.clear();
	aim_positions.resize(deform_graph.nodes.size());
	#pragma omp parallel for
	for (int i = 0; i < deform_graph.nodes.size(); i++){
		aim_positions[i] = deform_graph.nodes[i].Position;
	}
}


void sibr::GaussianView::UpdateCenterRadius(){
	for(int i = 0; i < deform_graph.indices_blocks.size(); i++){
		vector<unsigned int> cur_block = deform_graph.indices_blocks[i];
		Pos center(0.0, 0.0, 0.0);

		for (unsigned int j = 0; j < cur_block.size(); j++) {
			center += deform_graph.nodes[cur_block[j]].Position;
		}
		center = center / cur_block.size();
		aim_centers[i] = center;
	}
	return;
}

