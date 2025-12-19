cmake_minimum_required(VERSION 3.18)

project(ember LANGUAGES C CXX)

if(NOT APPLE)
    enable_language(CUDA)
    # Optional: Let CMake auto-detect the best GPU architecture for the host
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

option(EMBER_INPLACE_BUILD "Write built extension modules into ember/_core for local dev" OFF)
set(EMBER_PYCORE_DIR "${CMAKE_SOURCE_DIR}/ember/_core")

# Helper, filers CUDA for windows and linux and pure C for macos
macro(filter_backend_sources source_list)
    if(APPLE)
        list(FILTER ${source_list} EXCLUDE REGEX "_cuda\\.cu$")
    else()
        list(FILTER ${source_list} EXCLUDE REGEX "_cpu\\.c$")
    endif()
endmacro()

# ----------------------------------------------------
# Core Library
# ----------------------------------------------------
file(GLOB_RECURSE EMBER_COMMON_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/core/*.*"
)

filter_backend_sources(EMBER_COMMON_SOURCES)

add_library(ember_native STATIC ${EMBER_COMMON_SOURCES})
set_target_properties(ember_native PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ember_native PUBLIC "${CMAKE_SOURCE_DIR}/src")

if(NOT APPLE)
    set_target_properties(ember_native PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ----------------------------------------------------
# Extension Function
# ----------------------------------------------------
function(add_ember_extension target_name output_basename module_dir entry_file)
  if(NOT EXISTS "${module_dir}/${entry_file}")
    message(FATAL_ERROR "Entry file not found: ${module_dir}/${entry_file}")
  endif()

  # grab everything
  file(GLOB_RECURSE MOD_SOURCES CONFIGURE_DEPENDS
    "${module_dir}/*.c"
    "${module_dir}/*.cc"
    "${module_dir}/*.cpp"
    "${module_dir}/*.cu"
  )

  # filter out correct sources for given architecture
  filter_backend_sources(MOD_SOURCES)

  # make library
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

  # CUDA flags
  if(NOT APPLE)
    set_target_properties(${target_name} PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
  endif()
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