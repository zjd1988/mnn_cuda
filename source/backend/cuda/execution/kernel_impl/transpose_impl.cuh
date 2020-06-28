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

#ifndef __TRANSPOSE_IMPL_CUH__
#define __TRANSPOSE_IMPL_CUH__
#include <cuda_runtime.h>
#include "backend/cuda/core/runtime/CUDARuntime.hpp"

#define TRANSPOSE_MAX_DIMENSION 10
void CallTranspose(const int size, const void* input, const int* input_shape, const int* input_axis, const int shape_size,
                  void* output, MNN::CUDARuntime *runtime, MNN::DataType data_type);

#endif  // __TRANSPOSE_IMPL_CUH__
