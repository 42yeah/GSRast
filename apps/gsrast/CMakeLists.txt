add_executable(gsrast main.cpp GSRastWindow.cpp PointCloudShader.cpp GSPointCloud.cpp)
target_link_libraries(gsrast PUBLIC base)

add_subdirectory(shaders)
