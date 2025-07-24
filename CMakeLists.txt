cmake_minimum_required(VERSION 3.14)
project(AI)

include(FetchContent)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)

FetchContent_MakeAvailable(Eigen)

add_executable(main main.cpp layer.cpp loss.cpp)

target_include_directories(main PUBLIC ${Eigen3_INCLUDE_DIRS})

target_link_libraries(main Eigen3::Eigen)