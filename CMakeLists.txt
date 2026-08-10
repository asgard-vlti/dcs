cmake_minimum_required(VERSION 3.16)

project(dcs)

add_subdirectory(baldr_jcr)

add_subdirectory(commander)

find_package(PkgConfig)

add_library(ImageStreamIO)

target_sources(ImageStreamIO
    PRIVATE
        libImageStreamIO/ImageStreamIO.c
        # libImageStreamIO/ImCreate_ft.c
        # libImageStreamIO/ImCreate_test.c
    
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            libImageStreamIO
        FILES
            libImageStreamIO/ImageStreamIO.h
            libImageStreamIO/ImageStruct.h
            # libImageStreamIO/ImageStreamIOError.h
            libImageStreamIO/commander_structs.h

)

