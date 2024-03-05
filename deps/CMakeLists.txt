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

set(IMPLOT_SRC implot/implot.cpp implot/implot_demo.cpp implot/implot_items.cpp)
add_library(implot ${IMPLOT_SRC})
target_include_directories(implot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/implot)
target_link_libraries(implot PUBLIC imgui)

set(LMDB_SRC lmdb/libraries/liblmdb/mdb.c
    lmdb/libraries/liblmdb/midl.c)
add_library(lmdb ${LMDB_SRC})
target_include_directories(lmdb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lmdb/libraries/liblmdb)

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/stb)
