cmake_minimum_required(VERSION 3.5.0)

project(gsrast)

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

add_subdirectory(deps)
add_subdirectory(shaders)

set(SOURCES WindowBase.cpp
    Window.cpp
    DrawBase.cpp
    BasicWindow.cpp
    BufferGeo.cpp
    ShaderBase.cpp
    SimpleShader.cpp
    CameraBase.cpp
    FirstPersonCamera.cpp
    BBox.cpp
    Config.cpp
    RenderTarget.cpp
    Framebuffer.cpp)

add_executable(basic main.cpp ${SOURCES})
target_link_libraries(basic PUBLIC
    glfw
    glm::glm
    glad)

add_library(base ${SOURCES})
target_include_directories(base INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(base PUBLIC
    glfw
    glm::glm
    glad)

add_subdirectory(apps)
