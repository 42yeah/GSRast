add_executable(fbtest main.cpp FBTestWindow.cpp ColorfulShader.cpp)
target_link_libraries(fbtest PUBLIC base imgui lmdb)

add_subdirectory(shaders)
