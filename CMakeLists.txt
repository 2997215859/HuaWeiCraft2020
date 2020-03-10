cmake_minimum_required(VERSION 3.14)
project(HuaWeiCraft2020)

set(CMAKE_CXX_STANDARD 14)

add_definitions(-D OFFLINE)
add_definitions(-D TEST)

add_executable(HuaWeiCraft2020
        main.cpp)
