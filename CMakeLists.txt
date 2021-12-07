cmake_minimum_required(VERSION 3.16)
project(summa)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
include_directories("${MPI_INCLUDE_PATH}")
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

add_executable(summa main.cpp utils.cpp summa.cpp)
# target_compile_definitions(summa PRIVATE DEBUG=1)