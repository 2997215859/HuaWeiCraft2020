cmake_minimum_required(VERSION 3.14)
project(HuaWeiCraft2020)

set(CMAKE_CXX_STANDARD 11)

add_definitions(-D TEST )
add_definitions(-D NO_NEON)
add_definitions(-D ZCONF)

find_package(Threads REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread")



add_executable(hw_km KM/main.cpp )
target_link_libraries(hw_km Threads::Threads)
