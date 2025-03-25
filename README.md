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
|---transforms_train.json
```

### File Descriptions
*point_cloud.ply*, *cfg_args*, and *transforms_train.json* 


```bash
---point_cloud_config.txt

```

```bash
---graph.obj

```

```bash
---deform.txt

```

### Creating Your Own Data


## Interactive ARAP Deformation Usage

### Launching the Interactive System
   ```bash
   ./install/bin/SIBR_gaussianViewer_app -m path_to_your_dataset
   ```

### Mouse & Keyboard Controls

#### Basic Operations
| Key | Function |
|-----|----------|
| W | Move forward |
| A | Move left |
| S | Move backward |
| D | Move right |
| Space | Jump |
| Shift | Sprint |
| Esc | Exit / Cancel |


#### Mouse + Keyboard Combinations
| Keys | Function |
|------|----------|
| Alt + Right Click + Drag | Rotate around a point |
| Shift + Right Click + Drag | Fine-tune rotation |
| Ctrl + Middle Click + Drag | Precise panning |




### Example: Performing ARAP Deformation

#### Stage I Geometrical Deformation of Gaussians

#### Stage II Align 3DGS with Radiance Field

### Saving and Reproducing the Deformation Process


## TODO

Currently, this project supports ARAP deformation of radiance fields under the flat-Gaussian representation proposed in the paper *GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting*. We will provide detailed usage instructions as soon as possible.

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

