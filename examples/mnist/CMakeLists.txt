cmake_minimum_required(VERSION 3.0)

PROJECT(mnistex)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++0x")

option(MNISTEXAMPLE  "MNIST Example" OFF)
IF(${MNISTEXAMPLE})
    MESSAGE(STATUS "MNIST Example activated")
    FILE(GLOB_RECURSE  MNIST_EX_INC       *.h)
    FILE(GLOB_RECURSE  MNIST_EX_SRC       *.cpp)
    FILE(GLOB_RECURSE  MNIST_EX_UIS       *.ui)

    find_package(Qt5Widgets REQUIRED)
    find_package(Qt5Charts REQUIRED)

    find_package(MNIST)
    if(NOT MNIST_FOUND)
        message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/eidelen/mnist")
    endif(NOT MNIST_FOUND)

    SET(CMAKE_AUTOMOC ON)
    QT5_WRAP_UI(MNIST_EX_FORMS_HEADERS ${MNIST_EX_UIS} )
    include_directories( .  ${MNIST_INCLUDE_DIR} ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR} dataInput )
    add_executable(mnistex ${MNIST_EX_SRC} ${MNIST_EX_FORMS_HEADERS} )
    target_link_libraries(mnistex Qt5::Widgets Qt5::Charts pthread eidnnlib )

    # Pass MNIST data directory to main.cpp
    target_compile_definitions(mnistex PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
ENDIF()






