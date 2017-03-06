cmake_minimum_required(VERSION 3.0)

PROJECT(eidnnlib)

# enable c++ 11
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++0x")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

INCLUDE_DIRECTORIES(inc)
FILE(GLOB_RECURSE  EIDNN_INCLUDES        inc/*.h)
FILE(GLOB_RECURSE  EIDNN_SOURCES         src/*.cpp)

ADD_LIBRARY(eidnnlib SHARED ${EIDNN_INCLUDES} ${EIDNN_SOURCES} )
#TARGET_LINK_LIBRARIES(${LIBRARY_NAME} ${NATK_LIBRARY} )

target_compile_features(eidnnlib PRIVATE cxx_range_for)


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

