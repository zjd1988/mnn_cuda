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

#ifndef __UNARY_IMPL_H__
#define __UNARY_IMPL_H__

#include "backend/cuda/core/runtime/CUDARuntime.hpp"

enum CudaUnaryOpType {
    CudaUnaryOpOperation_ABS = 0,
    CudaUnaryOpOperation_NEG = 1,
    CudaUnaryOpOperation_FLOOR = 2,
    CudaUnaryOpOperation_CEIL = 3,
    CudaUnaryOpOperation_SQUARE = 4,
    CudaUnaryOpOperation_SQRT = 5,
    CudaUnaryOpOperation_RSQRT = 6,
    CudaUnaryOpOperation_EXP = 7,
    CudaUnaryOpOperation_LOG = 8,
    CudaUnaryOpOperation_SIN = 9,
    CudaUnaryOpOperation_COS = 10,
    CudaUnaryOpOperation_TAN = 11,
    CudaUnaryOpOperation_ASIN = 12,
    CudaUnaryOpOperation_ACOS = 13,
    CudaUnaryOpOperation_ATAN = 14,
    CudaUnaryOpOperation_RECIPROCAL = 15,
    CudaUnaryOpOperation_LOG1P = 16,
    CudaUnaryOpOperation_BNLL = 17,
    CudaUnaryOpOperation_ACOSH = 18,
    CudaUnaryOpOperation_SINH = 19,
    CudaUnaryOpOperation_ASINH = 20,
    CudaUnaryOpOperation_ATANH = 21,
    CudaUnaryOpOperation_SIGN = 22,
    CudaUnaryOpOperation_ROUND = 23,
    CudaUnaryOpOperation_COSH = 24,
    CudaUnaryOpOperation_ERF = 25,
    CudaUnaryOpOperation_ERFC = 26,
    CudaUnaryOpOperation_ERFINV = 27,
    CudaUnaryOpOperation_EXPM1 = 28,
    CudaUnaryOpOperation_SIGMOID = 29,
    CudaUnaryOpOperation_TANH = 30,
    CudaUnaryOpOperation_MIN = CudaUnaryOpOperation_ABS,
    CudaUnaryOpOperation_MAX = CudaUnaryOpOperation_TANH    
};

void callUnary(void *input, void *output, size_t count, MNN::CUDARuntime* runtime, MNNCUDADataType_t data_type, CudaUnaryOpType op_type);

#endif  // __UNARY_IMPL_H__
