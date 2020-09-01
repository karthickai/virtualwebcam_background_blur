cmake_minimum_required(VERSION 3.16)
project(virtual_webcam_bg_blur)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig)
set(OpenCV_DIR /opt/opencv/lib/cmake)
find_package(OpenCV REQUIRED )

#using pkg-config to getting Gstreamer
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

#including GStreamer header files directory
include_directories(
        ${GLIB_INCLUDE_DIRS}
        ${GSTREAMER_INCLUDE_DIRS}
)

#linking GStreamer library directory
link_directories(
        ${GLIB_LIBRARY_DIRS}
        ${GSTREAMER_LIBRARY_DIRS}
)

#using pk-config to getting TensorFlow packages
find_package(TensorflowCC REQUIRED)
add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${GSTREAMER_LIBRARIES} TensorflowCC::TensorflowCC)
find_package(PROTOBUF)
if(PROTOBUF_FOUND)
    message("protobuf found")
    target_include_directories(${PROJECT_NAME} PUBLIC ${PROTOBUF_INCLUDE_DIRS})
endif()
# You may also link cuda if it is available.
find_package(CUDA)
if(CUDA_FOUND)
    message("cuda found")
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${PROTOBUF_LIBRARIES})
endif()