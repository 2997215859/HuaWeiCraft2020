cmake_minimum_required(VERSION 3.14)
project(HuaWeiCraft2020)

set(CMAKE_CXX_STANDARD 11)

add_definitions(-D TEST )
add_definitions(-D NO_NEON)

add_executable(hw_lr LR/main.cpp)
add_executable(hw_svr SVR/main.cpp)
