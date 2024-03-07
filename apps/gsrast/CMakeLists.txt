add_executable(gsrast main.cpp
    GSRastWindow.cpp
    PointCloudShader.cpp
    GSPointCloud.cpp
    GSEllipsoids.cpp
    SplatShader.cpp
    GSGaussians.cpp
    CudaBuffer.cpp
    CopyShader.cpp
    DebugCamera.cpp
    SplatData.cpp
    Inspector.cpp
    RenderSelector.cpp
    PLYExplorer.cpp)

find_package(CUDAToolkit)

add_subdirectory(gscuda)

target_link_libraries(gsrast PUBLIC
    base
    # CudaRasterizer
    CUDA::cudart
    gscuda
    imgui
    implot
    lmdb
    stb)

add_subdirectory(shaders)
