cmake_minimum_required(VERSION 3.0)

PROJECT(eidnnlib)

# enable c++ 11
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++0x")

set(CMAKE_INCLUDE_CURRENT_DIR ON)


find_package(Eigen3 REQUIRED)
include_directories(EIGEN3_INCLUDE_DIR)

INCLUDE_DIRECTORIES(inc)
FILE(GLOB_RECURSE  EIDNN_INCLUDES        inc/*.h)
FILE(GLOB_RECURSE  EIDNN_SOURCES         src/*.cpp)

ADD_LIBRARY(eidnnlib SHARED ${EIDNN_INCLUDES} ${EIDNN_SOURCES} )
target_include_directories(eidnnlib INTERFACE inc)

option(TESTEIDNN  "TEST" OFF)
IF(${TESTEIDNN})
    MESSAGE(STATUS "Tests activated")
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})

    FILE(GLOB_RECURSE  EIDNN_TESTS_INC       test/*.h)
    FILE(GLOB_RECURSE  EIDNN_TESTS_SRC       test/*.cpp)

    add_executable(runTests ${EIDNN_TESTS_INC} ${EIDNN_TESTS_SRC})
    target_link_libraries(runTests ${GTEST_LIBRARIES} pthread eidnnlib )
ENDIF()






