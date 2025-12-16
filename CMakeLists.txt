cmake_minimum_required(VERSION 3.18)
project(ember LANGUAGES C CXX CUDA)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

option(EMBER_INPLACE_BUILD "Write built extension modules into ember/_core for local dev" OFF)
set(EMBER_PYCORE_DIR "${CMAKE_SOURCE_DIR}/ember/_core")

file(GLOB_RECURSE EMBER_COMMON_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/core/*.*"
)

add_library(ember_native STATIC ${EMBER_COMMON_SOURCES})
set_target_properties(ember_native PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ember_native PUBLIC "${CMAKE_SOURCE_DIR}/src")

function(add_ember_extension target_name output_basename module_dir entry_file)
  if(NOT EXISTS "${module_dir}/${entry_file}")
    message(FATAL_ERROR "Entry file not found: ${module_dir}/${entry_file}")
  endif()

  file(GLOB_RECURSE MOD_SOURCES CONFIGURE_DEPENDS
    "${module_dir}/*.c"
    "${module_dir}/*.cc"
    "${module_dir}/*.cpp"
    "${module_dir}/*.cu"
  )

  Python_add_library(${target_name} MODULE ${MOD_SOURCES} WITH_SOABI)

  set_target_properties(${target_name} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "${output_basename}"
  )

  if(EMBER_INPLACE_BUILD)
    set_target_properties(${target_name} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${EMBER_PYCORE_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${EMBER_PYCORE_DIR}"
    )
  endif()

  target_link_libraries(${target_name} PRIVATE Python::Module ember_native)
  target_include_directories(${target_name} PRIVATE "${CMAKE_SOURCE_DIR}/src")
endfunction()

add_ember_extension(_tensor "_tensor" "${CMAKE_SOURCE_DIR}/src/tensor" "tensor_api.c")
# add_ember_extension(_nn     "_nn"     "${CMAKE_SOURCE_DIR}/src/nn"     "nn_api.c")
# add_ember_extension(_optim  "_optim"  "${CMAKE_SOURCE_DIR}/src/optim"  "optim_api.c")

install(TARGETS 
    _tensor 
    # _nn 
    # _optim
  LIBRARY DESTINATION ember/_core
  RUNTIME DESTINATION ember/_core
)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/ember/"
  DESTINATION ember
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.pyi"
    PATTERN "py.typed"
)
