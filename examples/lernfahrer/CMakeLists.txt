cmake_minimum_required(VERSION 3.0)

PROJECT(lernfahrer)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++0x")

option(LERNFAHRER  "Lernfahrer Example" OFF)
IF(${LERNFAHRER})
    MESSAGE(STATUS "Lernfahrer example activated")
    FILE(GLOB  LERNFAHRER_EX_INC       *.h)
    FILE(GLOB  LERNFAHRER_EX_SRC       *.cpp)

    find_package(Qt5Widgets REQUIRED)

    SET(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    include_directories( .  ${LERNFAHRER_INCLUDE_DIR} ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR} )
    add_executable(lernfahrer ${LERNFAHRER_EX_SRC} resources.qrc )
    target_link_libraries(lernfahrer Qt5::Widgets pthread eidnnlib )

    option(TESTLERNFAHRER  "TEST LERNFAHRER" OFF)
    IF(${TESTLERNFAHRER})
        MESSAGE(STATUS "Tests Lernfahrer activated")
        find_package(GTest REQUIRED)
        include_directories(${GTEST_INCLUDE_DIRS})

        FILE(GLOB_RECURSE  LF_TESTS_INC       test/*.h)
        FILE(GLOB_RECURSE  LF_TESTS_SRC       test/*.cpp)

        add_executable(runLernfahrerTests ${LF_TESTS_INC} ${LF_TESTS_SRC} car.cpp car.h)
        target_link_libraries(runLernfahrerTests ${GTEST_LIBRARIES} Qt5::Widgets pthread eidnnlib )
        target_compile_features(runLernfahrerTests PRIVATE cxx_std_17 )
    ENDIF()

ENDIF()






