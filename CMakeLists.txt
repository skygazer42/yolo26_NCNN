cmake_minimum_required(VERSION 3.10)

project(yolo26_ncnn CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(YOLO26_BUILD_SEG "Build YOLO26 segmentation demo" ON)
option(YOLO26_BUILD_TOOLS "Build helper tools" ON)

find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)

add_library(yolo26
    src/yolo26.cpp
    src/yolo26_draw.cpp
    src/yolo26_preprocess.cpp
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
    )
    target_include_directories(yolo26_seg
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(yolo26_seg
        PUBLIC
            yolo26
    )

    add_executable(yolo26_seg_demo src/yolo26_seg_demo.cpp)
    target_link_libraries(yolo26_seg_demo yolo26_seg)
endif()

if(YOLO26_BUILD_TOOLS)
    add_executable(yolo26_topk_parity tools/topk_parity.cpp)
    target_include_directories(yolo26_topk_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

    add_executable(yolo26_mask_parity tools/mask_parity.cpp)
    target_include_directories(yolo26_mask_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(yolo26_mask_parity PRIVATE ncnn ${OpenCV_LIBS})
endif()
