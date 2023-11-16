cmake_minimum_required(VERSION 3.15)

project(HelloOpenGL)

set(TARGET_NAME HelloOpenGL)
FILE(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/*.cpp)

#enable cuda
add_executable(${TARGET_NAME} main.cpp)

enable_language(CUDA)
find_package(CUDA REQUIRED)
# set CUDA_NVCC_FLAGS
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_86,code=sm_86;-std=c++17;)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} ${CUDA_LIBRARIES})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_86,code=sm_86;-std=c++17;)

if(MSVC)
    add_custom_command(TARGET ${TARGET_NAME}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:external_libs> $<TARGET_FILE_DIR:${TARGET_NAME}>
                       COMMAND_EXPAND_LISTS)
    install(FILES $<TARGET_FILE:external_libs>
            DESTINATION ${TEST_INSTALL_PATH})
endif()

target_link_libraries(${TARGET_NAME} dependencies)
target_include_directories(${TARGET_NAME} PUBLIC ${TARGET_INCLUDE_DIRECTORIES})
