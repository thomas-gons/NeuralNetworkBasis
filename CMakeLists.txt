cmake_minimum_required(VERSION 3.14)
project(AI)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_POLICY_DEFAULT_CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)


FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type: Debug, Release, RelWithDebInfo, MinSizeRel" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx2 -ffast-math -march=native")

# check if mold is available
find_program(MOLD_EXECUTABLE mold)
if(MOLD_EXECUTABLE)
  message(STATUS "Using mold linker")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=mold")
endif()

add_executable(main
  src/main.cpp src/model.cpp src/layer.cpp src/loss.cpp
  src/utils/dataset.cpp src/utils/misc.cpp
)

target_compile_options(main PRIVATE -fexec-charset=UTF-8)

include(CheckIPOSupported)
check_ipo_supported(RESULT result)
if(result)
  set_property(TARGET main PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
endif()

# Ajoute les chemins vers les en-têtes
target_include_directories(main PUBLIC
    ${xtensor_INCLUDE_DIRS}
    ${xtensor-blas_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Lie les bibliothèques OpenBLAS et LAPACK (LE POINT CRUCIAL)
target_link_libraries(main PRIVATE
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    nlohmann_json::nlohmann_json
)