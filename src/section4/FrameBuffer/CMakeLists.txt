cmake_minimum_required(VERSION 3.15)

project(FrameBuffer)

set(TARGET_NAME FrameBuffer)
FILE(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/*.cpp)

#enable cuda
add_executable(${TARGET_NAME} main.cpp)

enable_language(CUDA)
target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-code arch=compute_86,code=[sm_86,compute_86] -std=c++17>)

if(MSVC)
    foreach(dep glfw)
        add_custom_command(TARGET ${TARGET_NAME}
                           POST_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                               $<TARGET_FILE:${dep}> 
                               $<TARGET_FILE_DIR:${TARGET_NAME}>)
    endforeach()
endif()

# opengl
target_link_libraries(${TARGET_NAME} dependencies)
target_include_directories(${TARGET_NAME} PUBLIC ${TARGET_INCLUDE_DIRECTORIES})
