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
#ifndef GAUSSIAN_VIEW_H
#define GAUSSIAN_VIEW_H
#pragma once

#include "helper.hpp"
#include "Deform.hpp"
#include "Samples.hpp"
#include "cudakdtree.cuh"
#include "Snapshot.hpp"

namespace CudaRasterizer
{
	class Rasterizer;
}

namespace sibr { 

	class BufferCopyRenderer;
	class BufferCopyRenderer2;

	/**
	 * \class RemotePointView
	 * \brief Wrap a ULR renderer with additional parameters and information.
	 */
	class SIBR_EXP_ULR_EXPORT GaussianView : public sibr::ViewBase
	{
		SIBR_CLASS_PTR(GaussianView);

	public:

		/**
		 * Constructor
		 * \param ibrScene The scene to use for rendering.
		 * \param render_w rendering width
		 * \param render_h rendering height
		 */
		GaussianView(const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h, const char* file, bool* message_read, int sh_degree, bool white_bg = false, bool useInterop = true, int device = 0);

		/** Replace the current scene.
		 *\param newScene the new scene to render */
		void setScene(const sibr::BasicIBRScene::Ptr & newScene);

		/**
		 * Perform rendering. Called by the view manager or rendering mode.
		 * \param dst The destination rendertarget.
		 * \param eye The novel viewpoint.
		 */
		void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;

		/**
		 * Update inputs (do nothing).
		 * \param input The inputs state.
		 */
		void onUpdate(Input& input, const Viewport & vp) override;

		/**
		 * Update the GUI.
		 */
		void onGUI() override;

		void RenderHelpers(const sibr::Viewport& viewport) override;

		/** \return a reference to the scene */
		const std::shared_ptr<sibr::BasicIBRScene> & getScene() const { return _scene; }

		void initgsHelperShader(void);

		void initGraphShader(void);

		void initMeshShader(void);

		void getGraphMesh();

		void	onRender(const Viewport & vpRender);

		void	ResetAll();

		void	ResetControls();

		virtual ~GaussianView() override;

		void init(const sibr::BasicIBRScene::Ptr & ibrScene, uint render_w, uint render_h, const char* file, bool* messageRead, int sh_degree, bool white_bg, bool useInterop, int device);

		bool* _dontshow;

		bool show_error;

		const char* _file;
		bool _white_bg, _useInterop;
		bool _is_record = true;
		bool flag_test_scale;

		bool _finish_rendering = false;

		std::vector<Pos> pos_vector;
		std::vector<Rot> rot_vector;
		std::vector<Vertex> vertices;
		std::vector<float> opacity_vector;
		std::vector<SHs<3>> shs_vector;
		std::vector<Scale> scale_vector;

		std::vector<EndPoint> ends_vector;
		std::vector<EndPoint> ends_vector_backup;

		std::vector<Vertex> ends_vertices;
		float end_coeff = 2.0f; 
		float axis_padding = 1e-3f;

		float ortho_scale = 2.0f;
		bool _enable_alpha = true;

		std::vector<int> gs_idx_in_pair;
		std::vector<int> sp_idx_in_pair;

		std::vector<float> half_length_vector;
		std::vector<Eigen::Matrix3f> sigma_vector;

		std::vector<Pos> pos_backup_vector;
		std::vector<Rot> rot_backup_vector;
		std::vector<float> opacity_backup_vector;
		std::vector<SHs<3>> shs_backup_vector;
		std::vector<Scale> scale_backup_vector;

		std::vector<int> indicies;

		std::vector<Scale> scale_3d_clip;
		std::vector<float> scale_3d_max;

		std::vector<Eigen::Matrix3f> shs_rot_vector;

		int global_sp_idx;
		vector<bool> global_gs_idx;

		vector<unsigned int> local_gs_idx;

		int selected_gs_idx;
		vector<bool> influenced_sp_idx;

		vector<float> transparance_for_test;
		bool flag_update_sp;
		bool flag_clip_containing;
		bool show_all_samples;
		bool additional_samples = true;

		PointPlusPayload* samples_pos_payload_gpu;
		vector<PointPlusPayload> samples_pos_payload_vector;


		std::vector<bool> too_small;

		std::vector<float> opacity_orig_vector;
		std::vector<Scale> scale_orig_vector;

		DeformGraph deform_graph;
		vector<Pos>    aim_centers;
		vector<float>  aim_centers_radius;
		vector<Pos>    aim_positions;

		DeformHistory history;
		int add_block_op_idx = 0;
		int mouse_move_op_idx = 0;
		bool run_history = false;
		int history_step = 0;
		int cur_move_step = 0;
		bool run_move = false;

		bool run_script = false;
		int script_step = 0;

		bool enable_adaptive_lpf = true;

		vector<Pos> drag_history;
		int playback_steps = 0;

		AABB aabb_overall;
		SamplePoints sps;
		float x_step, y_step, z_step, grid_step;
		float init_grid_step;
		float init_sample_step;
		float3 min_xyz;
		int grid_count;
		int grid_num = 64;
		int valid_grid_num;
		int samples_per_grid_dim, samples_count_per_grid;
		vector<int> valid_grid_idx;
		vector<int> grid_gs_prefix_sum_vector;
		int num_samples_per_dim;
		int total_estimated_containings;

		vector<Pos> sample_positions;
		vector<int> sample_neighbors_num;
		vector<Pos> backup_sample_positions;
		vector<SHs<3>> sample_feature_shs;
		vector<SHs<3>> sample_grad_shs;
		vector<float> sample_grad_opacity;

		vector<Pos> sample_positions_new;
		vector<Pos> backup_sample_positions_new;

		vector<SHs<3>> test_aim_shs;
		vector<SHs<3>> aim_feature_shs;
		vector<float> aim_opacity;
		vector<SHs<3>> result_grad_sign;

		vector<SHs<3>> dF_dshs_vector;
		vector<float> dF_dopacity_vector;

		vector<float> feature_opacity_vector;


		vector<int> samples_state;
		vector<vector<int>> gs_containing;

		vector<float> loss_vector;

    	set<unsigned int> static_indices;
    	set<unsigned int> control_indices;

		vector<vector<unsigned int>> indices_blocks;

		vector<int> blocks_type;	// 0 for free & un-selected
									// 1 for free & selected
									// -1 for static
		bool multi_control;

		int block_num;

		int max_x, max_y, min_x, min_y, lastX, lastY, xoffset, yoffset, press_x, press_y;
		bool flag_select, flag_rect, flag_show_nodes, flag_show_gs, flag_show_centers, flag_perform_shs_rot, flag_show_samples;

		bool show_single;
		int single_gaussian_index;

		vector<int> covering_gs_idx;

		int _estimated_coeff = 128;
		int _estimated_const = 200;
		std::string configfile;

		float _show_alpha_cutoff = 0.0f;
		float _additional_sample_threshold = 0.08f;

		//for the selected sample
		float _current_alpha = 0.0f;
		float _current_r = 0.0f;
		float _current_g = 0.0f;
		float _current_b = 0.0f;

		float _aim_alpha = 0.0f;
		float _aim_r = 0.0f;
		float _aim_g = 0.0f;
		float _aim_b = 0.0f;

		float _bg_color[3] = {1.0f, 1.0f, 1.0f};

		bool _to_ortho = false;
		bool _log_camera = false;


		Matrix4f T;

		Vector2f window_size;

		Vector4f screen_right, screen_up, screen_forward;

		std::shared_ptr<Mesh>		quadMesh;
		GLShader					gsHelperShader; ///< Overlay shader.
		GLParameter					ratiogsHelper2Dgpu; ///< Aspect ratio.
		GLParameter					gsHelperStateGPU; ///< Trackball state 

		std::shared_ptr<Mesh>		graphMesh;
		GLShader					GraphShader;
		GLParameter					PVM;
		GLParameter					MVP;
		GLParameter					Tex;
		GLuint						textureId;
		Image<unsigned char, 3> 	imageData;

		GLShader					MeshShader;

		int is_synthetic = 0;
		int has_soup = 0;
		bool high_quality = false;
		int has_simplified = 0;

		std::shared_ptr<Mesh>		centersMesh;
		std::shared_ptr<Mesh>		samplesMesh;
		std::shared_ptr<Mesh> 		EllipseMesh;
		std::shared_ptr<Mesh> 		SphereMesh;

		std::shared_ptr<MaterialMesh>		MeshGT = std::shared_ptr<MaterialMesh>(new MaterialMesh(true));
		std::shared_ptr<MaterialMesh>		Meshbackup = std::shared_ptr<MaterialMesh>(new MaterialMesh(true));

		std::vector<Vertex> mesh_vertices;
		std::vector<Vertex> soup_vertices;
		std::vector<Vertex> simplified_vertices;

		std::vector<Vertex_m> v_sphere;
		std::vector<Normal_m> n_sphere;
		std::vector<Face_m> f_sphere;

		std::vector<Snapshot> snapshots;
		Snapshot last_snapshot;

		std::vector<int> static_samples;
		std::vector<bool> static_gaussians;

		bool shadersCompiled;
		bool show_aim;
		int show_count;

		int node_num = 150;

		float degrees = 45.0f;
		
		bool nodes_on_mesh = false;

		std::vector<Pos> simplified_points;
		std::vector<Pos> simplified_points_backup;

		std::vector<Pos> mesh_points;
		std::vector<Pos> mesh_points_backup;

		std::vector<Pos> soup_points;
		std::vector<Pos> soup_points_backup;

		std::vector<Vertex> cand_vertices;
		std::vector<Pos> cand_points;
		std::vector<Pos> cand_points_backup;

		bool one_step_deform = false;
		bool _log_moved_ones = false; // log the nums of moved gs and sp

		std::vector<int> empty_grid;
		int* empty_grid_cuda;

		std::vector<Node> temp_aim_nodes;

		void setupWeights(const DeformGraph& dm);
		void setupWeightsforSamples(const DeformGraph& dm);
		void setupWeightsforMesh(const DeformGraph& dm);
		void setupWeightsforSoup(const DeformGraph& dm);
		void setupWeightsforEnds(const DeformGraph& dm);
		void UpdateIndicies(DeformGraph& dm);
		void UpdateAimPosition(Pos delta_pos, bool record_history);
		void UpdateAimPositionTwist(Vector4f axis, int y, bool record_history);
		void UpdateAimPositionScale(int y);
		void UpdatePosition(const DeformGraph &dm);
		void UpdatePositionforSamples(const DeformGraph &dm);
		void getQuadMesh();
		void getCentersMesh();
		void getSamplesMesh();
		void getOverallAABB();
		std::vector<SamplePoint> getGridSamples();
		void GPUSetupSamplesFeatures();
		void GPUOptimize();
		void CopyFeatureFromCPU2GPU();
		void CopyFeatureFromGPU2CPU();
		void RefreshAdam();
		void UpdateFeatures();
		void SetupUniformSamples(std::vector<SamplePoint>& current_sps);
		void TakeSnapshot();
		void LoadSnapshot(int idx);
		void UpdateGsCoverRange();
		void RebuildGraph();
		void RecordDeformation();
		void LoadDeformation();
		void UpdateAsSixPointsWithdrawBad(const DeformGraph &dm);
		void FastUpdateSamplesSH(const DeformGraph &dm);
		void RunHistoricalDeform();
		void CleanDeformHistory();
		void GetEndPoints();
		void UpdateContainingRelationship();
		void GetAdaLpfRatio();
		void UpdateCenterRadius();
		void JudgeEmptyGrid();
		void CopyPartialInfoFromGPU2CPU();
		void ExpandOpRange();
		void CheckStaticSamples(); //as well as static gaussians
		void CheckMovedGaussians();
		void LoadMeshForGraph();
		void ReloadAimPositions();
		void LoadDeformScript0();
		void LoadDeformScript1();
		void RunDeformScript();
		void UpdateContainingRelationshipWithStaticGrids();
		void LoadDeformScript2();
		void LoadDeformScript3();
		void LoadDeformScript4();
		void LoadDeformScript5();
		void LoadDeformScript6();
		void LoadDeformation_wo_rebuild();
		void LoadBackgroundGaussians();


	protected:

		std::string currMode = "Splats";

		bool _cropping = false;
		sibr::Vector3f _boxmin, _boxmax, _scenemin, _scenemax;
		char _buff[512] = "cropped.ply";

		bool _fastCulling = true;
		int _device = 0;
		int _sh_degree = 3;
		bool _optimize_options[12] = {true, true, true, true, true, true, false, true, true, true, false, true};
		float _learning_rate[5] = {0.0f, 0.0020f, 0.0050f, 0.1f, 0.0001f};
		bool _deform_options[4] = {true, true, true, true};
		int _optimize_steps = 101;
		bool* opt_options_cuda;
		float* learning_rate_cuda;
		float* ada_lpf_ratio;
		int _snap_idx = 0;
		int cur_step = 0;
		int _snapshot_interval = 100;
		int _containing_relation_interval = 10;
		int _kdtree_query_interval = 1;
		bool _constraints_on_center = false;
		bool _is_twist = false;
		bool _is_scale = false;
		bool _surface_graph = false;
		int _max_displacement_per_step = 50;
		bool _only_deform_gs = false;
		bool _only_deform_pos = false;
		char _deform_filepath[512] = "_deform.txt";
		std::string pcd_filepath;
		int _enforced_deform_steps = 10;
		int _rest_deform_steps = 0;
		Pos _displacement = Pos(0.0f, 0.0f, 0.02f);
		bool _record_history = true;
		bool _drag_direction;
		std::string input_deform_path;
		std::string input_graph_path;
		std::string compare_ply_path;
		int padding = 1;
		bool _show_deform_efficiency = false;

		bool fix_x = false;
		bool fix_y = false; 
		bool fix_z = false;
		
		bool _optimize_from_start = false;

		float _w_rot = 1.0f;
		float _w_reg = 10.0f;
		float _w_con = 100.0f;

		float _weighting_factor = 1.0f;

		bool _during_deform = false;
		int _reset_convergency_interval = 50;

		bool adjust_op_range = true;

		int count;
		float* pos_cuda;
		float* rot_cuda;
		float* scale_cuda;
		float* opacity_cuda;
		float* shs_cuda;

		float* shs_highlight_cuda;

		float* opacity_orig_cuda;
		float* scale_orig_cuda;

		float* scale_3d_max_gpu;
		float* cur_opacity_cuda;
		float* aim_opacity_cuda;

		float lpf_parameter = 0.2f;
		
		float* aim_feature_cuda;
		float* cur_feature_cuda;
		float* feature_grad_cuda;
		float* opacity_grad_cuda;
		float* dF_dopacity_cuda;
		float* dF_dshs_cuda;
		float* dF_dpos_cuda;
		float* dF_drot_cuda;
		float* dF_dscale_cuda;
		float* dF_dcov3D;

		float* m_opacity_cuda;
		float* v_opacity_cuda;
		float* m_shs_cuda;
		float* v_shs_cuda;
		float* m_pos_cuda;
		float* v_pos_cuda;
		float* m_rot_cuda;
		float* v_rot_cuda;
		float* m_scale_cuda;
		float* v_scale_cuda;

		int* static_samples_cuda;

		int* grid_index_cuda;
		int* gaussian_index_cuda;
		int* per_gs_grids_cuda;

		float low_pass_filter_param;

		vector<Scale> max_scale_vector;
		float* max_scale_cuda;

		vector<Eigen::Matrix3f> ada_lpf_vector;
		vector<int> gs_init_grid_idx;
		int* gs_init_grid_idx_cuda;

		bool flag_valid_deform = false;

		vector<int> new_gs_idx;

		vector<int> moved_gaussians;
		int* moved_gaussians_cuda;
		vector<int> current_static_grids;
		int* current_static_grids_cuda;


		float* total_feature_loss;
		float* total_shape_loss;
		int* rect_cuda;
		int* step;
		int* aim_index_cuda;

		int* grid_gs_prefix_sum_cuda;
		int* valid_grid_cuda;
		vector<int> grided_gs_idx;
		int* grided_gs_idx_cuda;
		bool* grid_is_converged_cuda;
		bool* grid_nearly_converged_cuda;
		float* grid_loss_sums_cuda;

		float* scale_3d_max_cuda;

		float* sample_neighbor_weights;
		int* sample_neighbor_nodes;
		Eigen::Quaternionf* q_vector_cuda;

		std::vector<int> is_surface_vector;
		int* surface_cuda;
		int _surface_count = 0;

		std::vector<int> gs_type_vector;
		int* gs_type_cuda;

		float* sample_pos_cuda;

		int* paired_gs_idx;
		int* paired_sp_idx;

		float* half_length_cuda;
		float* sigma_cuda;
		float* sigma_damp_cuda;


		int* gs_containing_prefix_sum;
		int* gs_containing_maximum_offset;
		int* neighbouring_sp_idx;

		GLuint imageBuffer;
		cudaGraphicsResource_t imageBufferCuda;

		size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
		void* geomPtr = nullptr, * binningPtr = nullptr, * imgPtr = nullptr;
		std::function<char* (size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;

		float* view_cuda;
		float* proj_cuda;
		float* cam_pos_cuda;
		float* background_cuda;

		float _scalingModifier = 1.0f;
		GaussianData* gData;

		bool _interop_failed = false;
		std::vector<char> fallback_bytes;
		float* fallbackBufferCuda = nullptr;
		bool accepted = false;

		float loss;
		float shape_loss;
		float lr_pos;

		int deform_mode = 0;

		int knn_k = 12;
		int new_knn_k = 12;

		DeformScript deform_script;

		bool show_background = true;
		bool has_background = false;
		vector<int> background_idx;
		string background_path;

		std::vector<Pos> bkg_pos;
		std::vector<Rot> bkg_rot;
		std::vector<float> bkg_opacity;
		std::vector<SHs<3>> bkg_shs;
		std::vector<Scale> bkg_scale;
		int bkg_count;

		float* all_pos_cuda;
		float* all_rot_cuda;
		float* all_opacity_cuda;
		float* all_shs_cuda;
		float* all_scale_cuda;

		std::shared_ptr<sibr::BasicIBRScene> _scene; ///< The current scene.
		PointBasedRenderer::Ptr _pointbasedrenderer;
		BufferCopyRenderer* _copyRenderer;
		GaussianSurfaceRenderer* _gaussianRenderer;
	};

} /*namespace sibr*/ 


#endif
