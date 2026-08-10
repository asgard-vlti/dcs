cmake_minimum_required(VERSION 3.16)

project(dcs)

add_subdirectory(baldr_jcr)

add_subdirectory(commander)

find_package(PkgConfig)

add_library(ImageStreamIO)

target_include_directories(ImageStreamIO
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/libImageStreamIO
)

target_sources(ImageStreamIO
    PRIVATE
        libImageStreamIO/ImageStreamIO.c
        # libImageStreamIO/ImCreate_ft.c
        # libImageStreamIO/ImCreate_test.c

)

