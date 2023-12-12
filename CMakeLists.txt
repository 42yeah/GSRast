cmake_minimum_required(VERSION 3.5.0)

project(gsrast)

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

add_subdirectory(deps)
add_subdirectory(shaders)

add_executable(gsrast main.cpp WindowBase.cpp Window.cpp DrawBase.cpp GSRasterWindow.cpp BufferGeo.cpp ShaderBase.cpp SimpleShader.cpp)
target_link_libraries(gsrast PUBLIC
    glfw
    glm::glm
    glad)
