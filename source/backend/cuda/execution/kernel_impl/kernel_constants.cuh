/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __KERNEL_CONSTANTS_CUH__
#define __KERNEL_CONSTANTS_CUH__
#include <map>
#include <string>
#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include "Type_generated.h"

// // Used by Pooling and Conv2d
// static constexpr char kSamePadModeUpperCase[] = "SAME";

// // Used by Pooling and Conv2d
// static constexpr char kSamePadModeLowerCase[] = "same";

// // Used by Pooling and Conv2d
// static constexpr char kValidPadModeUpperCase[] = "VALID";

// // Used by Pooling and Conv2d
// static constexpr char kValidPadModeLowerCase[] = "valid";

// // Used by Pooling
// static constexpr char kAvgPoolingModeUpperCase[] = "AVG";

// // Used by Pooling
// static constexpr char kAvgPoolingModeLowerCase[] = "avg";

// Used by MaxPool pad: The minimum value of float32
static constexpr float kSignedMinFloat = -3.402823466e+38F;

// Used by mixprecision, cudnn dtype select
static std::map<MNN::DataType, cudnnDataType_t> kCudnnDtypeMap = {{MNN::DataType_DT_FLOAT, CUDNN_DATA_FLOAT},
                                                                {MNN::DataType_DT_HALF, CUDNN_DATA_HALF},
                                                                {MNN::DataType_DT_INT32, CUDNN_DATA_INT32}};

static std::map<MNN::DataType, int> kCudnnDtypeLenMap = {{MNN::DataType_DT_FLOAT, sizeof(float)},
                                                                {MNN::DataType_DT_HALF, sizeof(int16_t)}, //len of half type equal to int16 
                                                                {MNN::DataType_DT_INT32, sizeof(int32_t)}};
// Used by mixprecision, cuda dtype select
static std::map<MNN::DataType, cudaDataType_t> kCudaDtypeMap = {{MNN::DataType_DT_FLOAT, CUDA_R_32F},
                                                              {MNN::DataType_DT_HALF, CUDA_R_16F}};


#endif  // __KERNEL_CONSTANTS_CUH__
