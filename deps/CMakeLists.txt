# Dependencies

add_library(glad glad/src/glad.c)
target_include_directories(glad PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/glad/include)

add_subdirectory(diff-gaussian-rasterization)

set(IMGUI_SRC imgui/imgui.cpp imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_opengl3.cpp)
add_library(imgui ${IMGUI_SRC})
target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
target_link_libraries(imgui PUBLIC glfw)
