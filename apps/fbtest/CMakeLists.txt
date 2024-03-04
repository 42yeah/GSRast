add_executable(fbtest main.cpp FBTestWindow.cpp ColorfulShader.cpp SamplerShader.cpp)
target_link_libraries(fbtest PUBLIC base imgui)

add_subdirectory(shaders)
