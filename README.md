project for person reid
===

    cp config.cfg.example config.cfg
    source init.sh
    
    
### cuda accelerated kcf
    /caffe/src/caffe/cukcf/fast.cpp
    /caffe/include/caffe/cukcf/fast.hpp
    /caffe/tools/finalrun.cpp
    # help math functions
    /caffe/src/caffe/util/math_functions.cu
    # build needs cufft lib
