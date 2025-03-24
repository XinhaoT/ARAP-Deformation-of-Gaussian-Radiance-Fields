# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr

set(SIBR_PROJECT "gaussian")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
project(sibr_${SIBR_PROJECT} LANGUAGES C CXX CUDA)

sibr_gitlibrary(TARGET CudaRasterizer
    GIT_REPOSITORY 	"https://github.com/XinhaoT/CudaRasterizer.git"
    GIT_TAG			"96ea96c97c9348d5987a37711210b03c07c28cf4"
)

find_package(CUDAToolkit REQUIRED)

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp" "*.cu" "*.cuh")
source_group("Source Files" FILES ${SOURCES})

file(GLOB SHADERS "shaders/*.frag" "shaders/*.vert" "shaders/*.geom")
source_group("Source Files\\shaders" FILES ${SHADERS})

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp" "*.cu" "*.cuh" "shaders/*.frag" "shaders/*.vert" "shaders/*.geom")

# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# set_source_files_properties(SOURCES PROPERTIES LANGUAGE CUDA)

enable_language("CUDA")
find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
set(CMAKE_CUDA_ARCHITECTURES 75)

# set(CMAKE_PREFIX_PATH /home/gaps-sccg/anaconda3/lib/python3.10/site-packages/torch/share/cmake/Torch)


## Specify target rules
add_library(${PROJECT_NAME} SHARED ${SOURCES})

################################
set(Torch_DIR /home/gaps-sccg/Downloads/libtorch-shared-with-deps-1.12.0+cu113/libtorch/share/cmake/Torch)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# find_package(Torch REQUIRED)



################################
target_include_directories(${PROJECT_NAME}
							PRIVATE
							${CMAKE_SOURCE_DIR}/include)

include_directories(${Boost_INCLUDE_DIRS} .)
if (WIN32)
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${ASSIMP_LIBRARIES}
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
	${OpenCV_LIBRARIES}
	# OpenMP::OpenMP_CXX
	glfw3
	sibr_system
	sibr_view
	sibr_assets
	sibr_renderer
	sibr_basic
	CUDA::cudart
	CudaRasterizer
)
else()
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${ASSIMP_LIBRARIES}
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
	${OpenCV_LIBRARIES}
	# OpenMP::OpenMP_CXX
	glfw
	sibr_system
	sibr_view
	sibr_assets
	sibr_renderer
	sibr_basic
	CUDA::cudart
	CudaRasterizer
)
endif()

# target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
# target_link_libraries(${PROJECT_NAME}  ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} )

# set_target_properties(
# 		${PROJECT_NAME}
#         PROPERTIES
#         CUDA_SEPARABLE_COMPILATION ON)

add_definitions( -DSIBR_EXP_ULR_EXPORTS -DBOOST_ALL_DYN_LINK  )

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "projects/${SIBR_PROJECT}/renderer")

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

## High level macro to install in an homogen way all our ibr targets
include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB                         ## mean install also MSVC IDE *.pdb file (DEST according to target type)
	SHADERS ${SHADERS}
	RSC_FOLDER ${SIBR_PROJECT}

    #STANDALONE  ${INSTALL_STANDALONE}   ## mean call install_runtime with bundle dependencies resolution
    COMPONENT   ${PROJECT_NAME}_install ## will create custom target to install only this project
)
