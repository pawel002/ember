cmake_minimum_required(VERSION 3.24)
project(ember LANGUAGES C CXX)

# Build the CUDA backend when a CUDA compiler is available. Pass
# -DEMBER_CUDA=OFF to force the CPU backend on a machine that has one.
option(EMBER_CUDA "Build the CUDA backend if a CUDA compiler is found" ON)

if(EMBER_CUDA AND NOT APPLE)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler found. Enabling CUDA backend.")
    # Default to the local GPU; pass -DCMAKE_CUDA_ARCHITECTURES=... to build for
    # another one (or on a machine without a GPU). Must be set before
    # enable_language, which otherwise picks its own default.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(EMBER_USE_CUDA ON)
  else()
    message(WARNING "CUDA compiler NOT found. Falling back to CPU backend.")
    set(EMBER_USE_CUDA OFF)
  endif()
else()
  set(EMBER_USE_CUDA OFF)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# find numpy includes
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT NUMPY_INCLUDE_DIR)
  message(
    FATAL_ERROR
      "NumPy include directory not found. Please ensure numpy is installed in the python environment: ${Python_EXECUTABLE}"
  )
else()
  message(STATUS "Found NumPy include path: ${NUMPY_INCLUDE_DIR}")
endif()

macro(filter_backend_sources source_list)
  if(EMBER_USE_CUDA)
    list(FILTER ${source_list} EXCLUDE REGEX "_cpu\\.c$")
  else()
    list(FILTER ${source_list} EXCLUDE REGEX "_gpu\\.cu$")
  endif()
endmacro()

# ----------------------------------------------------
# Core Library
# ----------------------------------------------------
file(GLOB_RECURSE EMBER_COMMON_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/src/core/*.c" "${CMAKE_SOURCE_DIR}/src/core/*.cu")

filter_backend_sources(EMBER_COMMON_SOURCES)

add_library(ember_native STATIC ${EMBER_COMMON_SOURCES})
set_target_properties(ember_native PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ember_native PUBLIC "${CMAKE_SOURCE_DIR}/src")

if(EMBER_USE_CUDA)
  target_link_libraries(ember_native PRIVATE CUDA::cublas)
endif()

# ----------------------------------------------------
# Extension Function
# ----------------------------------------------------
function(add_ember_extension target_name output_basename module_dir entry_file)
  if(NOT EXISTS "${module_dir}/${entry_file}")
    message(FATAL_ERROR "Entry file not found: ${module_dir}/${entry_file}")
  endif()

  # grab everything
  file(
    GLOB_RECURSE
    MOD_SOURCES
    CONFIGURE_DEPENDS
    "${module_dir}/*.c"
    "${module_dir}/*.cc"
    "${module_dir}/*.cpp"
    "${module_dir}/*.cu")

  # filter out correct sources for given architecture
  filter_backend_sources(MOD_SOURCES)

  # make library
  python_add_library(${target_name} MODULE ${MOD_SOURCES} WITH_SOABI)

  set_target_properties(${target_name}
                        PROPERTIES PREFIX "" OUTPUT_NAME "${output_basename}")

  target_link_libraries(${target_name} PRIVATE Python::Module ember_native)
  target_include_directories(${target_name} PRIVATE "${CMAKE_SOURCE_DIR}/src"
                                                    ${NUMPY_INCLUDE_DIR})

  if(EMBER_USE_CUDA)
    target_link_libraries(${target_name} PRIVATE CUDA::cublas)
  endif()
endfunction()

add_ember_extension(_tensor "_tensor" "${CMAKE_SOURCE_DIR}/src/tensor"
                    "tensor_api.c")

install(
  TARGETS _tensor
  LIBRARY DESTINATION ember/_core
  RUNTIME DESTINATION ember/_core)
