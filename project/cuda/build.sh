#!/bin/bash
cmake ../../../ \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_STL=c++_static \
-DMNN_USE_LOGCAT=false \
-DMNN_BUILD_BENCHMARK=ON \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3

make -j4
