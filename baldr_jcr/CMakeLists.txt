add_executable(baldr)

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

find_package(nlohmann_json 3.12.0 REQUIRED)

target_link_libraries(baldr
    PRIVATE
        ImageStreamIO
        Commander
        b64
        zmq
        cfitsio
        fmt
        boost_program_options
        nlohmann_json::nlohmann_json
)
