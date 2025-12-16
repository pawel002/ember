cmake_minimum_required(VERSION 3.18)
project(my_engine LANGUAGES CXX CUDA)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# 1. Gather Source Files
# GLOB_RECURSE finds all .cu and .cpp files in src/
file(GLOB_RECURSE SOURCE_FILES 
    "src/*.cpp"
    "src/*.cu"
    "src/*.cu"
)

# create the Python Extnsion Module
add_library(my_engine_core MODULE ${SOURCE_FILES})

# include directories
target_include_directories(my_engine_core PRIVATE "${CMAKE_SOURCE_DIR}/src")

# link libs
target_link_libraries(my_engine_core PRIVATE Python3::Module)

# flags
if(MSVC)
    target_compile_options(my_engine_core PRIVATE /O2)
else()
    target_compile_options(my_engine_core PRIVATE -O3 -fPIC)
    target_compile_options(my_engine_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_75,code=sm_75>)
endif()

install(TARGETS my_engine_core DESTINATION my_engine)