cmake_minimum_required(VERSION 3.15)

project(MultipleLights)

set(TARGET_NAME MultipleLights)
FILE(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/*.cpp)

#enable cuda
add_executable(${TARGET_NAME} main.cpp)

enable_language(CUDA)
find_package(CUDA REQUIRED)
# set CUDA_NVCC_FLAGS
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_86,code=sm_86;-std=c++17;)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} ${CUDA_LIBRARIES})

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
