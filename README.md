# ARAP-Deformation-of-Gaussian-Radiance-Fields

## Description

This is an implementation of the paper *As-Rigid-As-Possible Deformation of Gaussian Radiance Fields* [[paper]](https://doi.ieeecomputersociety.org/10.1109/TVCG.2025.3555404). It includes an interactive system that allows users to perform ARAP deformation on Gaussian Radiance Fields interactively.

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
The data should be formatted as follows before being used as input for the interactive system.

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
*point_cloud.ply*, *cfg_args*, and *transforms_test.json* are commonly used input and output file formats in the [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) project.


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
| `grid_num_per_dim`  | 64           | Number of spatial hashing grids per dimension. A sampling resolution of 64×64×64 is the default in our paper. Higher resolutions may yield slightly better results but reduce efficiency and may exceed GPU memory limits.|
| `is_synthetic`      | 1            | Indicates whether the dataset is synthetic. This parameter does not affect the algorithm's performance; it only indicates whether the dataset is synthetic (allowing quantitative metrics to be computed using ground truth).|
| `high_quality`      | 0            | For scenes with extremely high-frequency details, such as flat-Gaussian, enabling this setting may improve quality. For 3DGS representation, the default value is 0. See the implementation details in our paper for more information.  |

#### graph.obj (optinal)
This .obj file can be used as the specified graph structure for Embedded Deformation.

#### deform.txt (optinal)
This is a data file we defined to support exporting and importing the deformation process, allowing deformation recording and reproduction.

### Creating Your Own Data
Rebuild the Gaussian radiance field of your customized scene using the [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) project, and construct the dataset in the above format as input.



## Interactive System User Guide

### Launching the Interactive System
   ```bash
   ./install/bin/SIBR_gaussianViewer_app -m <path_to_your_dataset>
   ```

### Mouse & Keyboard Controls
The following is an introduction to the new features related to Gaussian radiance field deformation, built on the visualization tools of the [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [SIBR](https://sibr.gitlabpages.inria.fr/) projects, with the most common ones highlighted. 

#### Control Region Types Explaination
During deformation, three types of control regions are highlighted in different colors:


green - active region: the target position of the active region is moved by mouse dragging
yellow - pinned region： the region is intended to remain fixed at its original position during deformation
red - excluded region： the region does not participate in the deformation at all 


#### Mouse + Keyboard Combinations
| Keys | Function |
|------|----------|
| **`M` + Right Click + Drag** | drag to draw a box to add a control region |
| `K` + Right Click + Drag | drag to draw a box to set the selected control region as the active region |
| **`B` + Left Click + Drag** | drag to deform, moving the target position of the active region with the mouse |


#### Keyboard Operations
| Key | Function |
|-----|----------|
| **`N`** | switch the active region |
| **`J`** | set the current active region as excluded region |
| **`C`** | delete the current active region |
| `U` | set the current active as pinned region |
| **`V`** | switch the deform opertion type (bending/twisting/scaling) |
| `X` | change the target type to maintain during deformation: individual control nodes or the center of control nodes within each control region |
| `T` | clean the selected control regions |
| **`R`** | reset |
| `F1` | take a snapshot (of Gaussians and sampled radiance field) |
| `F2` | load a snapshot (with the index shown in Box `Load Snapshot`) |
| `<-` | load the previous snapshot |
| `->` | load the next snapshot |
| **`F6`** | optimize gaussians to align them with the radiance field (During optimization, the results at specified steps are saved as snapshots, which can be reviewed and compared using the `<-` and `->` keys) |
| `F` | show/hide Gaussian radiance field |
| **`G`** | show/hide Graph |
| **`H`** | show/hide highlights for control region |
| `F4` | show/hide the samples of radiance field |
| `F8` | switch between aim/current samples of radiance field |


#### Button Operations
| Button | Function |
|-----|----------|
| **`Load Mesh for Graph`** | load a mesh from an .obj file as the graph structure for deformation |
| **`Rebuild Deform Graph`** | reconstruct the graph structure based on the given number of control nodes |
| `Set new knn` | set the k for KNN with given number |
| **`Record Deformation`** | record all deformation processes and save them as a file |
| **`Load Deformation`** | read a deformation process file |
| **`Run Deformation`** | run the read deformation process |
| `Clean Deformation` | clear the existing deformation operations |
| `Load Deform Script[X]` | read the specified deformation operation script |
| `Run Deform Script` | run the deformation operation script |
| `Clear Snapshots` | clear all the snapshots |



### A Simple Example: Performing ARAP Deformation

#### Stage I Geometrical Deformation of Gaussians
1. Press `M` + Right Click + Drag. Do this operation mutiple times to select all the control regions you want.
2. Press `N` and `J` to choose the active region and the excluded region.
3. Press `B` + Left Click + Drag. Perform the customized deformation.

#### Stage II Align 3DGS with Radiance Field
4. Press `F6` to optimize the Gaussians to align them with the radiance field. 

### Saving and Reproducing the Deformation Process
After deformation, click the `Record Deformation` button to store the geometry deformation process.

#### Case I: Load the Stored Deformation Process
Reproduce the deformation process from the *pinocchio* example in the paper:
1. Set k value of KNN to 8.
2. Click the `Load Deformation` button to load the *deform.txt*
3. Click the `Run Deformation` button to perform geometrical deformation in Stage I, press `H` to hide the highlights for control regions
4. Press `F6` to optimize the Gaussians to align them with the radiance field 

#### Case II: Load the Deformation from Scipt (code)
Reproduce the deformation process from the *stripe* example in the paper:
1. Click the `Load Mesh for Graph` button, and load the *graph.obj*
2. Enable the `Build Graph on Mesh` box
3. Click the `Rebuild Deform Graph` button
2. Click the `Load Deform Script0` button
3. Click the `Run Deform Script` button, press `H` to hide the highlights for control regions
4. Press `F6` to optimize the Gaussians to align them with the radiance field

#### Case III: Load the Stored Deformation Process with Given Graph
1. Click the `Load Mesh for Graph` button, and load the *graph.obj*
2. Enable the `Build Graph on Mesh` box
3. Click the `Rebuild Deform Graph` button 
2. Click the `Load Deformation w/o Rebuild` button to load the *deform.txt*
3. Click the `Run Deformation` button to perform geometrical deformation in Stage I, press `H` to hide the highlights for control regions
4. Press `F6` to optimize the Gaussians to align them with the radiance field 

#### Case IV: Combo Deformation
Users can achieve the complex composite deformations in the supplementary materials of the paper by combining the above deformation methods with customized drag-based deformation.

#### NOTE
The k value in KNN is not stored in the deformation record file or the given deformation scripts. Note that different k values will result in different deformation outcomes. To reproduce the exact deformation process, ensure consistency in the k value.


## TODO
We will release the datasets presented in the paper (including .blend files and ground-truth images) soon. 

We will upload our video of this interactive system soon.

Currently, this project also supports ARAP deformation of radiance fields under the flat-Gaussian representation proposed in the paper *GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting*. We will provide detailed usage instructions as soon as possible.

## Acknowledgments

We would like to acknowledge [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [SIBR](https://sibr.gitlabpages.inria.fr/) as we have used parts of their codebase in our project.

## BibTeX

```bibtex

@ARTICLE{Tong2025arapGaussian,
author={Tong, Xinhao and Shao, Tianjia and Weng, Yanlin and Yang, Yin and Zhou, Kun},
journal={ IEEE Transactions on Visualization \& Computer Graphics },
title={{ As-Rigid-As-Possible Deformation of Gaussian Radiance Fields }},
year={2025},
doi={10.1109/TVCG.2025.3555404},
url = {https://doi.ieeecomputersociety.org/10.1109/TVCG.2025.3555404},
month=mar}

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

