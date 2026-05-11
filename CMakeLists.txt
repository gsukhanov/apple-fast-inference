cmake_minimum_required(VERSION 3.20)

project(fastinf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckCXXSourceCompiles)

find_library(ACCELERATE_FRAMEWORK Accelerate)

check_cxx_source_compiles("
    #include <arm_neon.h>
    #if !defined(__ARM_NEON) && !defined(__ARM_NEON__)
    #error NEON is not enabled for this target
    #endif
    int main() {
        int8x16_t value = vdupq_n_s8(0);
        return vgetq_lane_s8(value, 0);
    }
" FASTINF_HAS_NEON)

find_package(nlohmann_json REQUIRED)
find_package(benchmark REQUIRED)
find_package(OpenCV QUIET)
find_package(GTest REQUIRED)

add_library(${PROJECT_NAME}_objs INTERFACE)
target_compile_definitions(${PROJECT_NAME}_objs INTERFACE
    FASTINF_HAS_OPENCV=$<BOOL:${OpenCV_FOUND}>
    FASTINF_HAS_ACCELERATE=$<BOOL:${ACCELERATE_FRAMEWORK}>
    FASTINF_HAS_NEON=$<BOOL:${FASTINF_HAS_NEON}>
)
if(APPLE AND ACCELERATE_FRAMEWORK)
    target_compile_options(${PROJECT_NAME}_objs INTERFACE
        -Wno-deprecated-declarations
    )

    target_link_libraries(${PROJECT_NAME}_objs INTERFACE
        ${ACCELERATE_FRAMEWORK}
    )

    target_compile_definitions(${PROJECT_NAME}_objs INTERFACE
        ACCELERATE_NEW_LAPACK
    )
endif()
target_include_directories(${PROJECT_NAME}_objs INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(${PROJECT_NAME}_objs INTERFACE
    nlohmann_json::nlohmann_json
)
if(OpenCV_FOUND)
    target_include_directories(${PROJECT_NAME}_objs INTERFACE
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(${PROJECT_NAME}_objs INTERFACE
        ${OpenCV_LIBS}
    )
endif()

file(GLOB PROJECT_MAIN_SOURCE CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)
add_executable(${PROJECT_NAME} ${PROJECT_MAIN_SOURCE})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_objs)

file(GLOB_RECURSE PROJECT_TEST_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp
)
add_executable(${PROJECT_NAME}_unittest ${PROJECT_TEST_SOURCES})
get_target_property(GTEST_INCLUDE_DIRS GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)
if(GTEST_INCLUDE_DIRS)
    target_include_directories(${PROJECT_NAME}_unittest BEFORE PRIVATE
        ${GTEST_INCLUDE_DIRS}
    )
endif()
target_link_libraries(${PROJECT_NAME}_unittest PRIVATE
    ${PROJECT_NAME}_objs
    GTest::gtest_main
)

file(GLOB_RECURSE PROJECT_BENCH_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp
)
add_executable(${PROJECT_NAME}_benchmark ${PROJECT_BENCH_SOURCES})
target_link_libraries(${PROJECT_NAME}_benchmark PRIVATE ${PROJECT_NAME}_objs benchmark::benchmark_main)
