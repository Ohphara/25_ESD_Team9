cmake_minimum_required(VERSION 3.10)
project(yolo11n_tracker)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# 소스 파일 모으기
set(SRC
    main.cpp
    bytetrack/src/yolo11n.cpp
    bytetrack/src/BYTETracker.cpp
    bytetrack/src/kalmanFilter.cpp
    bytetrack/src/lapjv.cpp
    bytetrack/src/STrack.cpp
    bytetrack/src/bytetrack.cpp
    bytetrack/src/utils.cpp
)

# 헤더 포함 경로
include_directories(
    ./bytetrack/include
    /usr/local/include/ncnn
    ${CMAKE_SOURCE_DIR}/eigen-3.3.9
)

# OpenCV 찾기
find_package(OpenCV REQUIRED)

# ncnn 정적 라이브러리 지정
set(NCNN_LIB /usr/local/lib/ncnn/libncnn.a)

add_executable(yolo11n ${SRC})

target_link_libraries(yolo11n
    ${OpenCV_LIBS}
    ${NCNN_LIB}
)

# 이 줄이 실행파일을 루트 디렉토리에 생성!
set_target_properties(yolo11n PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
