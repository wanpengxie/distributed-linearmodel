project(distLM C CXX)

cmake_minimum_required(VERSION 3.4)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)
SET(Protobuf_USE_STATIC_LIBS ON)

# add_definitions('-g')
add_definitions('-O3')
add_definitions('-Wall')
add_definitions('-std=c++11')
add_definitions('-std=c++11')

find_package(Protobuf REQUIRED)
SET(config_path ${PROJECT_SOURCE_DIR}/include/conf)
execute_process(COMMAND protoc --proto_path=${config_path} --cpp_out=${config_path} ${config_path}/config.proto)

include_directories(
        "${PROJECT_SOURCE_DIR}/ps-lite/include"
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/src"
        "${PROTOBUF_INCLUDE_DIR}"
)

link_directories(
        "${PROJECT_BINARY_DIR}/ps-lite"
)

link_directories(
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_BINARY_DIR}/ps-lite"
)

add_subdirectory(ps-lite)

add_executable(main src/main.cpp ${config_path}/config.pb.cc)
target_link_libraries(main pslite)

add_subdirectory(tests)