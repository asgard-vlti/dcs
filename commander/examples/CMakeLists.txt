cmake_minimum_required(VERSION 3.18)

project(example VERSION 0.1.0 LANGUAGES CXX)

add_executable(example_server
    src/example.cpp
    src/gyro.cpp
)

add_executable(example_client
    src/client.cpp
)

target_include_directories(example_server PUBLIC include)
target_include_directories(example_client PUBLIC include)

target_compile_features(example_server PUBLIC cxx_std_17)
target_compile_features(example_client PUBLIC cxx_std_17)

target_link_libraries(example_server PUBLIC
    commander
)

target_link_libraries(example_client PUBLIC
    commander
)
