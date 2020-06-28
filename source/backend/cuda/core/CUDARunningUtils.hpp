//
//  OpenCLRunningUtils.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CUDARunningUtils_hpp
#define CUDARunningUtils_hpp

#include <string>
#include <vector>
#include <algorithm>

#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/cuda/core/runtime/CUDARuntime.hpp"

namespace MNN {
namespace CUDA {

inline std::vector<int> tensorShapeFormat(const Tensor *input) {
    int iN = (0 != input->buffer().dim[0].extent) ? input->buffer().dim[0].extent : 1;
    int iC = (0 != input->buffer().dim[1].extent) ? input->buffer().dim[1].extent : 1;
    int iH = (0 != input->buffer().dim[2].extent) ? input->buffer().dim[2].extent : 1;
    int iW = (0 != input->buffer().dim[3].extent) ? input->buffer().dim[3].extent : 1;

    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC) {
        iN = (0 < input->buffer().dim[0].extent) ? input->buffer().dim[0].extent : 1;
        iH = (0 < input->buffer().dim[1].extent) ? input->buffer().dim[1].extent : 1;
        iW = (0 < input->buffer().dim[2].extent) ? input->buffer().dim[2].extent : 1;
        iC = (0 < input->buffer().dim[3].extent) ? input->buffer().dim[3].extent : 1;
    }
    if (input->buffer().dimensions == 2) {
        iN = input->buffer().dim[0].extent;
        iH = 1;
        iW = 1;
        iC = input->buffer().dim[1].extent;
    }
    if (input->buffer().dimensions == 1) {
        iN = 1;
        iH = 1;
        iW = 1;
        iC = input->buffer().dim[0].extent;
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("tensorShapeFormat : [%d, %d, %d, %d] \n", iN, iH, iW, iC);
#endif
    std::vector<int> shape_vec{iN, iH, iW, iC};

    return shape_vec;
}

inline void *CUDABuffer(const Tensor *tensor) {
    return (void *)(tensor->deviceId());
}



} // namespace CUDA
} // namespace MNN
#endif  /* OpenCLRunningUtils_hpp */
