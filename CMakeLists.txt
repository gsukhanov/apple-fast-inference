cmake_minimum_required(VERSION 3.20)

project(fastinf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(nlohmann_json REQUIRED)
find_package(benchmark REQUIRED)

add_library(${PROJECT_NAME}_objs INTERFACE)
target_include_directories(${PROJECT_NAME}_objs INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(${PROJECT_NAME}_objs INTERFACE nlohmann_json::nlohmann_json)

file(GLOB PROJECT_MAIN_SOURCE CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)
add_executable(${PROJECT_NAME} ${PROJECT_MAIN_SOURCE})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_objs)

file(GLOB_RECURSE PROJECT_BENCH_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark/*.cpp
)
add_executable(${PROJECT_NAME}_benchmark ${PROJECT_BENCH_SOURCES})
target_link_libraries(${PROJECT_NAME}_benchmark PRIVATE ${PROJECT_NAME}_objs benchmark::benchmark_main)
