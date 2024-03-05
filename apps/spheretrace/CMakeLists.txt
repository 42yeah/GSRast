add_executable(spheretrace main.cpp Ellipsoid.cpp SphereTraceWindow.cpp SphereTraceShader.cpp)
target_link_libraries(spheretrace PUBLIC
    base
    imgui
    lmdb)

add_subdirectory(shaders)
