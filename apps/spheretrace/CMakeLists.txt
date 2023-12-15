add_executable(spheretrace main.cpp Cube.cpp SphereTraceWindow.cpp SphereTraceShader.cpp)
target_link_libraries(spheretrace PUBLIC
    base)

add_subdirectory(shaders)
