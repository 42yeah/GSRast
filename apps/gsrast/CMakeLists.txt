add_executable(gsrast main.cpp
    GSRastWindow.cpp
    PointCloudShader.cpp
    GSPointCloud.cpp
    GSEllipsoids.cpp
    SplatShader.cpp)
target_link_libraries(gsrast PUBLIC base)

add_subdirectory(shaders)
