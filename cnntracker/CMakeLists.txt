cmake_minimum_required(VERSION 2.8)
project( cnntracker )

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( /home/zhouyz14/caffe/include /home/zhouyz14/caffe/build/src )
include_directories( /usr/local/cuda/include )

link_directories ( /home/zhouyz14/caffe/build/lib  )
link_directories ( /usr/local/cuda/lib /usr/local/cuda/lib64 )

add_executable( cnntracker cnntracker )

target_link_libraries( cnntracker ${OpenCV_LIBS} caffe cuda cufft cblas)
