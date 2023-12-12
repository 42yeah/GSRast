set(SHADERS_DEST_DIR ${CMAKE_BINARY_DIR}/shaders)

add_custom_target(shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/simple ${SHADERS_DEST_DIR}/simple
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/orbital ${SHADERS_DEST_DIR}/orbital
    COMMENT "Deploying shaders...")
