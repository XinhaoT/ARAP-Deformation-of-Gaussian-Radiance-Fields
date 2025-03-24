# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr



project(SIBR_gaussianViewer_app LANGUAGES C CXX CUDA)

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp" "*.cu")
source_group("Source Files" FILES ${SOURCES})

file(GLOB RESOURCES "resources/*.ini")
source_group("Resources Files" FILES ${RESOURCES})

add_executable(${PROJECT_NAME} ${SOURCES})


# find_package(CUDAToolkit REQUIRED)
# enable_language("CUDA")
find_package(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75)
# include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# if(${CMAKE_BUILD_TYPE} STREQUAL "RELEASE")
#   set(NVCC_FLAGS -O3 -Wno-deprecated-gpu-targets)
#   message(STATUS "${PROJECT_NAME} compile cuda code in release mode")
# else()
#   set(NVCC_FLAGS -G -Wno-deprecated-gpu-targets)
#   message(STATUS "${PROJECT_NAME} compile cuda code in debug mode")
# endif()


						
# foreach (arch IN ITEMS ${CMAKE_CUDA_ARCHITECTURES})
# #    message("status adding arch=${arch}")
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
# endforeach()
# target_compile_options(SIBR_gaussianViewer_app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75>)
# target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)

target_include_directories(${PROJECT_NAME}
							PRIVATE
							${CMAKE_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME}

	${Boost_LIBRARIES}
	${ASSIMP_LIBRARIES}
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
	${OpenCV_LIBRARIES}
	OpenMP::OpenMP_CXX
	sibr_gaussian
	sibr_view
	sibr_assets
	sibr_renderer
	sibr_basic
)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "projects/gaussian/apps")

## High level macro to install in an homogen way all our ibr targets
include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB                         ## mean install also MSVC IDE *.pdb file (DEST according to target type)
	RESOURCES  	${RESOURCES}
	RSC_FOLDER 	"gaussian"
    STANDALONE  ${INSTALL_STANDALONE}   ## mean call install_runtime with bundle dependencies resolution
    COMPONENT   ${PROJECT_NAME}_install ## will create custom target to install only this project
)
