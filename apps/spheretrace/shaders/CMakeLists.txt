set(SHADERS_DEST_DIR ${CMAKE_BINARY_DIR}/shaders)

add_custom_target(st_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/st ${SHADERS_DEST_DIR}/st)
