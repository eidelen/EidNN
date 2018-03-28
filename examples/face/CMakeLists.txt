cmake_minimum_required(VERSION 3.0)

PROJECT(faceex)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++0x")

option(FACEEXAMPLE  "Face Example" OFF)
IF(${FACEEXAMPLE})
    MESSAGE(STATUS "Face Example activated")
    FILE(GLOB_RECURSE  FACE_EX_INC       *.h)
    FILE(GLOB_RECURSE  FACE_EX_SRC       *.cpp)
    FILE(GLOB_RECURSE  FACE_EX_UIS       *.ui)

    find_package(Qt5Widgets REQUIRED)
    find_package(Qt5Charts REQUIRED)
    find_package(OpenCV REQUIRED )

    SET(CMAKE_AUTOMOC ON)
    QT5_WRAP_UI(FACE_EX_FORMS_HEADERS ${FACE_EX_UIS} )
    include_directories( .  ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR} )
    add_executable(faceex ${FACE_EX_SRC} ${FACE_EX_FORMS_HEADERS} )
    target_link_libraries(faceex Qt5::Widgets Qt5::Charts ${OpenCV_LIBS} pthread eidnnlib )
ENDIF()






