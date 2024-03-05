set(SHADERS_DEST_DIR ${CMAKE_BINARY_DIR}/shaders)

add_custom_target(fbtest_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/colorful ${SHADERS_DEST_DIR}/colorful
    COMMENT "Deploying fbtest shaders..")
