set(SHADERS_DEST_DIR ${CMAKE_BINARY_DIR}/shaders)

add_custom_target(gsrast_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/pointcloud ${SHADERS_DEST_DIR}/pointcloud
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/splats ${SHADERS_DEST_DIR}/splats
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/splatsRef ${SHADERS_DEST_DIR}/splatsRef
    COMMENT "Deploying GSRast PC shaders..")
