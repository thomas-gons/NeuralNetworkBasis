cmake_minimum_required(VERSION 3.14)
project(AI)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_executable(main main.cpp layer.cpp loss.cpp)

target_include_directories(main PUBLIC ${Eigen3_INCLUDE_DIRS})

target_link_libraries(main Eigen3::Eigen)