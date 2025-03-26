# ARAP-Deformation-of-Gaussian-Radiance-Fields

## Description

This is an implementation of the paper *ARAP-Deformation-of-Gaussian-Radiance-Fields*. It includes an interactive system that allows users to perform ARAP deformation on Gaussian Radiance Fields interactively.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/XinhaoT/ARAP-Deformation-of-Gaussian-Radiance-Fields.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ARAP-Deformation-of-Gaussian-Radiance-Fields
   ```
3. Build the project using CMake:
   ```bash
   # Dependencies
   sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
   # build
   cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j24 --target install
   ```
4. **Note:** Modify `CMAKE_CUDA_ARCHITECTURES` in the CMake file according to your GPU architecture. The current setting is `75`, tested on an **RTX 2080 Super**. We have tested on an **RTX 4090** with `CMAKE_CUDA_ARCHITECTURES=89` as well.
5. **Attention:** During the build process, the required libraries will be downloaded automatically and stored in the `extlibs` folder. Before running the project, please ensure that all dependencies have been successfully downloaded.

## Quick Start

Run the following command to start the interactive application:

```bash
./install/bin/SIBR_gaussianViewer_app -m datasets/stripes/
```

### Optional

You can specify the number of OpenMP threads by setting the `OMP_NUM_THREADS` environment variable. For example:

```bash
OMP_NUM_THREADS=16 ./install/bin/SIBR_gaussianViewer_app -m datasets/stripes/
```


## Data Format and Preparation

### Dataset Directory Structure
```bash
<dataset_name>
|---point_cloud
|   |---iteration_<number>
|   |   |---point_cloud.ply
|   |   |---point_cloud_config.txt
|   |   |---graph.obj (optional)
|   |   |---deform.txt (optional)
|---cfg_args
|---transforms_test.json
```

### File Descriptions
*point_cloud.ply*, *cfg_args*, and *transforms_test.json* 


#### point_cloud_config.txt
The `point_cloud_config.txt` file follows a simple format where contains three constant in the order:

```
<grid_num_per_dim> <is_synthetic> <high_quality>
```

Default example:
```
64 1 0
```
This example sets `grid_num_per_dim` to 64, `is_synthetic` to 1 (true), and `high_quality` to 0 (false).

| Parameter            | Default Value | Description |
|----------------------|--------------|-------------|
| `grid_num_per_dim`  | 64           | Number of spatial hashing grids per dimension. |
| `is_synthetic`      | 1            | Indicates whether the dataset is synthetic.  |
| `high_quality`      | 0            | Determines whether high-quality mode is enabled, for flat-gaussian or  |

#### graph.obj (optinal)


#### deform.txt (optinal)


### Creating Your Own Data
[3DGS](https://github.com/graphdeco-inria/gaussian-splatting)

## Interactive ARAP Deformation Usage

### Launching the Interactive System
   ```bash
   ./install/bin/SIBR_gaussianViewer_app -m <path_to_your_dataset>
   ```

### Mouse & Keyboard Controls

#### Mouse + Keyboard Combinations
| Keys | Function |
|------|----------|
| Left Click + Drag | rotate the Scene |
| M + Right Click + Drag | draw a box to add one control region |
| K + Right Click + Drag | draw a box to set the ...|
| B + Left Click + Drag | drag to deform |


#### Keyboard Operations
| Key | Function |
|-----|----------|
| F | show/hide Gaussian radiance field |
| G | show/hide Graph |
| H | show/hide highlights for control region |
| F4 | show/hide the samples of radiance field |
| F8 | switch between aim/current samples of radiance field |
| N |  |
| J |  |
| C |  |
| U |  |
| V | switch the deform opertion type (bending/twisting/scaling) |
| X |  |
| T | clean the selected control regions |
| R | reset |
| F1 | take a snapshot (of Gaussians and sampled radiance field) |
| F2 | load a snapshot (with the index shown in ...) |
| <- | load the previous snapshot |
| -> | load the next snapshot |
| F6 | optimize gaussians to align them with the radiance field |



#### Control Region Explaination

```bash
green - 
yellow - 
red - 
```



### Example: Performing ARAP Deformation

#### Stage I Geometrical Deformation of Gaussians

#### Stage II Align 3DGS with Radiance Field

### Saving and Reproducing the Deformation Process

#### Case I:

#### Case II:

#### Case III:


## TODO
We will release the whole datasets (including .blend files and ground-truth images) soon. 

Currently, this project also supports ARAP deformation of radiance fields under the flat-Gaussian representation proposed in the paper *GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting*. We will provide detailed usage instructions as soon as possible.

## Acknowledgments

We would like to acknowledge [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [SIBR](https://sibr.gitlabpages.inria.fr/) as we have used parts of their codebase in our project.

## BibTeX

```bibtex
@Article{kerbl3Dgaussians,
  author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{"u}hler, Thomas and Drettakis, George},
  title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal      = {ACM Transactions on Graphics},
  number       = {4},
  volume       = {42},
  month        = {July},
  year         = {2023},
  url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

