# Dependencies

add_library(glad glad/src/glad.c)
target_include_directories(glad PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/glad/include)

# add_subdirectory(diff-gaussian-rasterization)
