add_executable(gsrast main.cpp
    GSRastWindow.cpp
    PointCloudShader.cpp
    GSPointCloud.cpp
    GSEllipsoids.cpp
    SplatShader.cpp
    GSGaussians.cpp
    CudaBuffer.cpp
    CopyShader.cpp
    DebugCamera.cpp)

find_package(CUDAToolkit)

add_subdirectory(gscuda)

target_link_libraries(gsrast PUBLIC
    base
    # CudaRasterizer
    CUDA::cudart
    gscuda)

add_subdirectory(shaders)
