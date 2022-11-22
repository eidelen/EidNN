cmake_minimum_required(VERSION 3.0)

PROJECT(eidnnlib)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


find_package(Eigen3 REQUIRED)

INCLUDE_DIRECTORIES(inc)
FILE(GLOB_RECURSE  EIDNN_INCLUDES        inc/*.h)
FILE(GLOB_RECURSE  EIDNN_SOURCES         src/*.cpp)

ADD_LIBRARY(eidnnlib SHARED ${EIDNN_INCLUDES} ${EIDNN_SOURCES} )
target_include_directories(eidnnlib INTERFACE inc)
target_link_libraries(eidnnlib Eigen3::Eigen )
target_compile_features(eidnnlib PRIVATE cxx_std_17 )

option(TESTEIDNN  "TEST" OFF)
IF(${TESTEIDNN})
    MESSAGE(STATUS "Tests activated")

    # Download and build gtest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    FILE(GLOB_RECURSE  EIDNN_TESTS_INC       test/*.h)
    FILE(GLOB_RECURSE  EIDNN_TESTS_SRC       test/*.cpp)

    add_executable(runTests ${EIDNN_TESTS_INC} ${EIDNN_TESTS_SRC})
    target_link_libraries(runTests eidnnlib gtest_main)
    target_compile_features(runTests PRIVATE cxx_std_17 )
ENDIF()






