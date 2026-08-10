cmake_minimum_required(VERSION 3.16)

project(dcs)

add_subdirectory(baldr_jcr)

add_subdirectory(commander)

find_package(PkgConfig)
find_package(Threads REQUIRED)

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

target_link_libraries(ImageStreamIO
    PUBLIC
        Threads::Threads
)

option(DCS_BUILD_LEGACY_MAKE_TARGETS "Build non-CMake dcs modules via legacy Makefiles" ON)

if(DCS_BUILD_LEGACY_MAKE_TARGETS)
    find_program(DCS_MAKE_EXECUTABLE NAMES make REQUIRED)

    # Single orchestration target so `cmake --build` also drives legacy Makefiles.
    add_custom_target(dcs_legacy_make_all ALL)
    add_dependencies(dcs_legacy_make_all baldr)

    # Legacy Makefiles expect these archives under dcs/lib.
    add_custom_target(dcs_legacy_stage_libs
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/lib
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:Commander>
                ${CMAKE_CURRENT_SOURCE_DIR}/lib/libcommander.a
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:ImageStreamIO>
                ${CMAKE_CURRENT_SOURCE_DIR}/lib/libImageStreamIO.a
        USES_TERMINAL
        VERBATIM
    )
    add_dependencies(dcs_legacy_stage_libs Commander ImageStreamIO)

    add_custom_target(dcs_make_baldr
        COMMAND ${DCS_MAKE_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR}/baldr clean all
        USES_TERMINAL
        VERBATIM
    )

    add_custom_target(dcs_make_baldr_tt
        COMMAND ${DCS_MAKE_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR}/baldr_tt clean all
        USES_TERMINAL
        VERBATIM
    )

    add_custom_target(dcs_make_heimdallr
        COMMAND ${DCS_MAKE_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR}/heimdallr clean all
                "CFLAGS=-std=c++17 -Wall -Wextra -ggdb -O1 -I../commander/include -I../libImageStreamIO -I/usr/local/include -I/usr/include/eigen3"
                "LDFLAGS=-L../lib -lcommander -lm -lzmq -lImageStreamIO -lb64 -lpthread -lboost_program_options -lfmt -lfftw3"
        USES_TERMINAL
        VERBATIM
    )

    add_dependencies(dcs_legacy_make_all
        dcs_legacy_stage_libs
        dcs_make_baldr
        dcs_make_baldr_tt
        dcs_make_heimdallr
    )
    add_dependencies(dcs_make_baldr dcs_legacy_stage_libs)
    add_dependencies(dcs_make_baldr_tt dcs_legacy_stage_libs)
    add_dependencies(dcs_make_heimdallr dcs_legacy_stage_libs)

    # Vendor SDK-dependent builds: only include when SDK paths are present.
    if(EXISTS "/opt/Boston Micromachines")
        add_custom_target(dcs_make_asgard_dm_server
            COMMAND ${DCS_MAKE_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR}/asgard-dm-server clean all
            USES_TERMINAL
            VERBATIM
        )
        add_dependencies(dcs_legacy_make_all dcs_make_asgard_dm_server)
    else()
        message(STATUS "Skipping asgard-dm-server legacy Makefile target: /opt/Boston Micromachines not found")
    endif()

    if(EXISTS "/opt/EDTpdv")
        add_custom_target(dcs_make_asgard_cred1_server
            COMMAND ${DCS_MAKE_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR}/asgard-cred1-server clean all
            USES_TERMINAL
            VERBATIM
        )
        add_dependencies(dcs_legacy_make_all dcs_make_asgard_cred1_server)
    else()
        message(STATUS "Skipping asgard-cred1-server legacy Makefile target: /opt/EDTpdv not found")
    endif()
endif()

