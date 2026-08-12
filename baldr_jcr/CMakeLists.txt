add_executable(baldr)

target_compile_features(baldr PRIVATE cxx_std_17)

target_compile_options(
    baldr
    PRIVATE
    -Wall -Wextra -Wpedantic -Wno-missing-field-initializers
    -DEIGEN_STACK_ALLOCATION_LIMIT=259200
    -save-temps
)

target_sources(baldr
    PRIVATE
        baldr.cpp
        servo_loop.cpp

    # PUBLIC
    #     FILE_SET HEADERS
    #     FILES
    #         baldr.h

)

find_package(nlohmann_json 3.7.3 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

target_link_libraries(baldr
    PRIVATE
        ImageStreamIO
        Commander
        b64
        zmq
        cfitsio
        fmt
        boost_program_options
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
)
