add_executable(gsrast main.cpp GSRastWindow.cpp OrbitalShader.cpp GSPointCloud.cpp)
target_link_libraries(gsrast PUBLIC base)
