cmake_minimum_required(VERSION 3.15)

project(cuda_opengl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(HOST_FLAGS "-O3")
    if(CMAKE_BUILD_TYPE STREQUAL Debug)
        set(HOST_FLAGS "-g")
    endif()
    set(HOST_FLAGS          "${HOST_FLAGS} -Wall -Wextra -Wfatal-errors -fexceptions -fPIC")
elseif(MSVC)
    set(HOST_FLAGS "-O2")
    if(CMAKE_BUILD_TYPE STREQUAL Debug)
        set(HOST_FLAGS "-DEBUG")
    endif()
    set(HOST_FLAGS "${HOST_FLAGS} -W4 -utf-8 -EHa")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) # TODO(ZhuFei): will export APIs only in the future
endif()
set(CUDA_FLAGS          "${HOST_FLAGS} -forward-unknown-to-host-compiler --extended-lambda --relocatable-device-code=true -Xcompiler")
set(CMAKE_C_FLAGS       "${HOST_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS     "${HOST_FLAGS} ${CMAKE_CXX_FLAGS}")

# add extern libraries
# find opengl
find_package(OpenGL REQUIRED)

# GLFW
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_INSTALL OFF CACHE BOOL " " FORCE)
set(GLFW_INSTALL OFF CACHE BOOL " " FORCE)
set(GLFW_USE_CHDIR OFF CACHE BOOL " " FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL " " FORCE)

add_subdirectory(extern/glfw)

mark_as_advanced(
    GLFW_BUILD_DOCS GLFW_BUILD_EXAMPLES GLFW_BUILD_INSTALL GLFW_BUILD_TESTS
    GLFW_DOCUMENT_INTERNALS GLFW_INSTALL GLFW_USE_CHDIR GLFW_USE_MENUBAR
    GLFW_USE_OSMESA GLFW_VULKAN_STATIC GLFW_USE_RETINA GLFW_USE_MIR
    BUILD_SHARED_LIBS USE_MSVC_RUNTIME_LIBRARY_DLL
)

# glad or glew
list(APPEND TARGET_INCLUDE_DIRECTORIES "extern/glfw/include" "extern/imgui")
if (MSVC)
    list(APPEND TARGET_INCLUDE_DIRECTORIES "extern/glad")
    list(APPEND TARGET_LIBRARIES opengl32 glad)
    add_subdirectory(extern/glad)
    set_property(TARGET glad PROPERTY FOLDER "extern/Renderer")
else()
    find_package(GLEW REQUIRED)
    list(APPEND TARGET_INCLUDE_DIRECTORIES ${GLEW_INCLUDE_DIRS})
    list(APPEND TARGET_LIBRARIES GL ${GLEW_LIBRARIES})
    list(APPEND TARGET_LIBRARIES GLEW::GLEW)
endif()

# JsonCpp
# 禁用 jsoncpp 的测试、示例等
set(JSONCPP_WITH_TESTS OFF CACHE BOOL "" FORCE)
set(JSONCPP_WITH_POST_BUILD_UNITTEST OFF CACHE BOOL "" FORCE)
set(JSONCPP_WITH_PKGCONFIG_SUPPORT OFF CACHE BOOL "" FORCE)
set(JSONCPP_WITH_CMAKE_PACKAGE OFF CACHE BOOL "" FORCE)
set(JSONCPP_WITH_EXAMPLE OFF CACHE BOOL "" FORCE)
mark_as_advanced(JSONCPP_WITH_TESTS JSONCPP_WITH_POST_BUILD_UNITTEST JSONCPP_WITH_PKGCONFIG_SUPPORT JSONCPP_WITH_CMAKE_PACKAGE JSONCPP_WITH_EXAMPLE)
add_subdirectory(extern/jsoncpp)
list(APPEND TARGET_INCLUDE_DIRECTORIES "extern/jsoncpp/include")
list(APPEND TARGET_LIBRARIES jsoncpp_lib jsoncpp_static jsoncpp_object)
set_property(TARGET jsoncpp_lib PROPERTY FOLDER "extern/JsonCpp")
set_property(TARGET jsoncpp_static PROPERTY FOLDER "extern/JsonCpp")
set_property(TARGET jsoncpp_object PROPERTY FOLDER "extern/JsonCpp")

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# IMGUI
# imgui
list(APPEND IMGUI_SOURCES
    extern/imgui/imgui.cpp
    extern/imgui/backends/imgui_impl_glfw.cpp
    extern/imgui/backends/imgui_impl_opengl3.cpp
    extern/imgui/imgui_draw.cpp
    extern/imgui/imgui_tables.cpp
    extern/imgui/imgui_widgets.cpp
)
list(APPEND TARGET_INCLUDE_DIRECTORIES extern/)
add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PRIVATE extern/imgui extern/imgui/backends ${TARGET_INCLUDE_DIRECTORIES})
set_property(TARGET imgui PROPERTY FOLDER "extern/Renderer")

# GLM
list(APPEND TARGET_INCLUDE_DIRECTORIES extern/glm)
add_subdirectory(extern/glm)
set_property(TARGET glm_static PROPERTY FOLDER "extern/Renderer")
set_property(TARGET glm_shared PROPERTY FOLDER "extern/Renderer")

list(APPEND TARGET_INCLUDE_DIRECTORIES include)

set(SHADER_PATH "${PROJECT_SOURCE_DIR}/shader")
set(ASSETS_PATH "${PROJECT_SOURCE_DIR}/assets")
configure_file(${CMAKE_SOURCE_DIR}/include/Config.h.ini ${CMAKE_SOURCE_DIR}/include/Config.h)


if(MSVC)
    add_subdirectory(shader)
else()
    find_package(assimp REQUIRED)
    list(APPEND TARGET_INCLUDE_DIRECTORIES ${ASSIMP_INCLUDE_DIRS})
    list(APPEND TARGET_LIBRARIES ${ASSIMP_LIBRARIES})
endif()

# Render base system
list(APPEND TARGET_INCLUDE_DIRECTORIES Engine/Renderer/include)
FILE(GLOB_RECURSE RENDER_SYS_SOURCE Engine/Renderer/src/*.cpp)
add_library(Renderer STATIC ${RENDER_SYS_SOURCE})
target_link_libraries(Renderer PUBLIC glfw imgui ${TARGET_LIBRARIES} ${OPENGL_LIBRARIES} glm::glm)
target_include_directories(Renderer PUBLIC ${TARGET_INCLUDE_DIRECTORIES})
set_property(TARGET Renderer PROPERTY FOLDER "extern/Renderer")

# cuda data structure
list(APPEND TARGET_INCLUDE_DIRECTORIES Engine/cuda)
list(APPEND TARGET_LIBRARIES Renderer glfw imgui glm::glm ${OPENGL_LIBRARIES})

# Create a library target (such as named "external_libs"), which includes all third-party and based defined librarie
add_library(dependencies INTERFACE)
target_link_libraries(dependencies INTERFACE ${TARGET_LIBRARIES})
# Add include directories for the library target
target_include_directories(dependencies INTERFACE ${TARGET_INCLUDE_DIRECTORIES} ${OPENGL_INCLUDE_DIR})

# Add src directory, which contains the source code of the project
add_subdirectory(src)