add_executable(gsrast main.cpp
    GSRastWindow.cpp
    PointCloudShader.cpp
    GSPointCloud.cpp
    GSEllipsoids.cpp
    SplatShader.cpp
    GSGaussians.cpp
    CudaBuffer.cpp)

find_package(CUDAToolkit)

target_link_libraries(gsrast PUBLIC
    base
    CudaRasterizer
    CUDA::cudart)

add_subdirectory(shaders)
