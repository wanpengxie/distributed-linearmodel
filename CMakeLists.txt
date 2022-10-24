project(distLM C CXX)

cmake_minimum_required(VERSION 3.4)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)

# add_definitions('-g')
add_definitions('-O3')
add_definitions('-Wall')
add_definitions('-std=c++11')
add_definitions('-std=c++11')

include_directories(
        "${PROJECT_SOURCE_DIR}/ps-lite/include"
        "${PROJECT_BINARY_DIR}/include"
        "${PROJECT_SOURCE_DIR}/src/conf"
)


link_directories(
        "${PROJECT_BINARY_DIR}/ps-lite"
)

add_subdirectory(ps-lite)

find_package(Protobuf REQUIRED)

add_executable(main src/main.cpp)

# Generate protobuf headers and sources

add_subdirectory(tests)
