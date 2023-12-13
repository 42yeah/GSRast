set(SHADERS_DEST_DIR ${CMAKE_BINARY_DIR}/shaders)

add_custom_target(gsrast_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/pointcloud ${SHADERS_DEST_DIR}/pointcloud
    COMMENT "Deploying GSRast PC shaders..")
