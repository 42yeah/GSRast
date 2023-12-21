add_executable(fbtest main.cpp FBTestWindow.cpp ColorfulShader.cpp SamplerShader.cpp)
target_link_libraries(fbtest PUBLIC base)

add_subdirectory(shaders)
