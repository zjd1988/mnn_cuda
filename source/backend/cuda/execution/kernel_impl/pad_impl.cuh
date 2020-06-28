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

#ifndef __PAD_IMPL_CUH__
#define __PAD_IMPL_CUH__
#include <cuda_runtime.h>
#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include "backend/cuda/execution/kernel_impl/kernel_constants.cuh"

void CallPad(const size_t size, const void* input, const int num, const int channels, const int old_height,
            const int old_width, const int padded_height, const int padded_width, const int pad_top, const int pad_left,
            float pad_value, void* output, MNN::CUDARuntime *runtime, MNN::DataType data_type);

#endif  // __PAD_IMPL_CUH__
