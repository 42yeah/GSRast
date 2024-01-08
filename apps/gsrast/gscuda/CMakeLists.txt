cmake_minimum_required(VERSION 3.20.0)
project(GSCuda CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75)

set(GSCUDA_SRC GSCuda.cu AuxBuffer.cu)
set_source_files_properties(${GSCUDA_SRC} PROPERTIES LANGUAGE CUDA)
add_library(gscuda ${GSCUDA_SRC})
target_include_directories(gscuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(glm REQUIRED)

target_link_libraries(gscuda PUBLIC
    glm::glm)
