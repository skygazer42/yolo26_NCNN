cmake_minimum_required(VERSION 3.10)

project(yolo26_ncnn CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(YOLO26_BUILD_SEG "Build YOLO26 segmentation demo" ON)

find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)

add_library(yolo26
    src/yolo26.cpp
    src/yolo26_draw.cpp
)

target_include_directories(yolo26
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(yolo26
    PUBLIC
        ncnn
        ${OpenCV_LIBS}
)

add_executable(yolo26_det src/yolo26_det.cpp)
target_link_libraries(yolo26_det yolo26)

if(YOLO26_BUILD_SEG)
    add_library(yolo26_seg
        src/yolo26_seg.cpp
        src/yolo26_seg_postprocess.cpp
    )
    target_include_directories(yolo26_seg
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(yolo26_seg
        PUBLIC
            yolo26
    )

    add_executable(yolo26_seg_demo src/yolo26_seg_demo.cpp)
    target_link_libraries(yolo26_seg_demo yolo26_seg)
endif()
