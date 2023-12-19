cmake_minimum_required(VERSION 3.15)

project(Shader)

file(GLOB_RECURSE SHADER_FILES "${CMAKE_SOURCE_DIR}/*.vs" "${CMAKE_SOURCE_DIR}/*.fs" "${CMAKE_SOURCE_DIR}/*.gs")

add_custom_target(Shaders SOURCES ${SHADER_FILES})

foreach(SHADER ${SHADER_FILES})
    get_filename_component(SHADER_PATH "${SHADER}" PATH)
    file(RELATIVE_PATH SHADER_PATH_REL "${CMAKE_SOURCE_DIR}/" "${SHADER_PATH}")
    string(REPLACE "/" "\\" GROUP_PATH "${SHADER_PATH_REL}")
    source_group("\\${GROUP_PATH}" FILES "${SHADER}")
endforeach()