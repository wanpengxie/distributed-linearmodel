#set(STA_DEPS protobuf)
add_executable(metric_test metric/metric_test.cpp)
target_include_directories(metric_test PUBLIC ${PROJECT_SOURCE_DIR}/include/metric ${PROJECT_SOURCE_DIR}/ps-lite/include)

add_executable(config_test conf/config_test.cpp ${PROJECT_SOURCE_DIR}/include/conf/config.pb.cc)
target_include_directories(config_test PUBLIC ${PROJECT_SOURCE_DIR}/include/conf ${PROJECT_SOURCE_DIR}/ps-lite/include)
target_link_libraries(config_test protobuf)

add_executable(io_test io/io_test.cpp)
target_include_directories(io_test PUBLIC ${PROJECT_SOURCE_DIR}/include/io)