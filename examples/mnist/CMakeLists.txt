cmake_minimum_required(VERSION 3.0)

PROJECT(mnistex)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++0x")

option(MNISTEXAMPLE  "MNIST Example" OFF)
IF(${MNISTEXAMPLE})
    MESSAGE(STATUS "MNIST Example activated")
    FILE(GLOB_RECURSE  MNIST_EX_INC       *.h)
    FILE(GLOB_RECURSE  MNIST_EX_SRC       *.cpp)
    FILE(GLOB_RECURSE  MNIST_EX_UIS       *.ui)
    include_directories( . ../../extern/mnist/include  ${CMAKE_BINARY_DIR} ) # use the external mnist-loader
    find_package(Qt5Widgets REQUIRED)
    SET(CMAKE_AUTOMOC ON)
    QT5_WRAP_UI(MNIST_EX_FORMS_HEADERS ${MNIST_EX_UIS} )
    add_executable(mnistex ${MNIST_EX_SRC} ${MNIST_EX_FORMS_HEADERS} )
    target_link_libraries(mnistex Qt5::Widgets pthread eidnnlib )
ENDIF()






