cmake_minimum_required(VERSION 3.14)
project(HuaWeiCraft2020)

set(CMAKE_CXX_STANDARD 11)

add_definitions(-D TEST )
add_definitions(-D NO_NEON)
add_definitions(-D ZCONF)

find_package(Threads REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread")


add_executable(hw_lr LR/main.cpp )
target_link_libraries(hw_lr Threads::Threads)


add_executable(hw_svr SVR/main.cpp)

add_executable(hw_bp BP/main.cpp)
target_link_libraries(hw_bp Threads::Threads)
