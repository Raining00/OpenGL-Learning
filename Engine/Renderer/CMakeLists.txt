cmake_minimum_required(VERSION 3.15)

project(Renderer)
if(!MSVC)
    find_package(CUDA REQUIRED)
endif()

set(TARGET Renderer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE RENDER_HEARD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB_RECURSE RENDER_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(${TARGET} STATIC ${RENDER_SOURCE_FILES})
target_link_libraries(${TARGET} PUBLIC glfw imgui ${TARGET_LIBRARIES} ${OPENGL_LIBRARIES} glm::glm)
target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/extern ${CMAKE_SOURCE_DIR}/extern/imgui ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

enable_language(CUDA)
target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                    --generate-code arch=compute_86,code=[sm_86,compute_86] -std=c++17>)
target_include_directories(${TARGET} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${TARGET} PRIVATE ${CUDA_LIBRARIES}) 


foreach(RENDER_FILE ${RENDER_HEARD_FILES} ${RENDER_SOURCE_FILES})
    get_filename_component(RENDER_PATH "${RENDER_FILE}" PATH)
    file(RELATIVE_PATH RENDER_PATH_REL "${CMAKE_CURRENT_SOURCE_DIR}/" "${RENDER_PATH}")
    string(REPLACE "/" "\\" GROUP_PATH "${RENDER_PATH_REL}")
    source_group("\\${GROUP_PATH}" FILES "${RENDER_FILE}")
endforeach()