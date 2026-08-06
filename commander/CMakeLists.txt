add_library(Commander)

target_compile_features(Commander PRIVATE cxx_std_17)

target_compile_options(Commander PRIVATE -Wall -Wextra -Wpedantic)


target_sources(Commander
    PRIVATE
        src/server/interactive.cpp
        src/server/single_command.cpp
        src/server/socket.cpp
        src/client/socket.cpp
        src/registry.cpp
        src/server.cpp
        src/module.cpp
        src/utils.cpp

    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            ./include
)
