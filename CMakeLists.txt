cmake_minimum_required(VERSION 3.18)
project(ember LANGUAGES C CXX CUDA)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

file(GLOB_RECURSE SOURCE_FILES "src/*.c" "src/*.cu")

add_library(_core MODULE ${SOURCE_FILES})

set_target_properties(_core PROPERTIES PREFIX "")
set_target_properties(_core PROPERTIES OUTPUT_NAME "_core")

target_link_libraries(_core PRIVATE Python3::Module)
target_include_directories(_core PRIVATE "${CMAKE_SOURCE_DIR}/src")

if(MSVC)
    target_compile_options(_core PRIVATE /O2)
else()
    target_compile_options(_core PRIVATE 
        $<$<COMPILE_LANGUAGE:C>:-O3 -fPIC>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -fPIC>
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_75,code=sm_75>
    )
endif()

install(TARGETS _core DESTINATION ember)

install(FILES src/bindings/_tensor.pyi 
        DESTINATION ember 
        RENAME _core.pyi)