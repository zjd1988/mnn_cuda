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

#include "backend/cuda/execution/kernel_impl/pad_impl.cuh"

template <typename T>
__global__ void Pad(const size_t size, const T* input, const int num, const int channels, const int old_height,
                    const int old_width, const int padded_height, const int padded_width, const int pad_top,
                    const int pad_left, float pad_value, T* output) {
    T pad_value_ = static_cast<T>(pad_value);
    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
        int block_num = pos / padded_width / padded_height;
        const int padded_w = pos % padded_width;
        const int padded_h = pos / padded_width % padded_height;
        if (padded_h - pad_top < 0 || padded_w - pad_left < 0 || padded_h - pad_top >= old_height ||
              padded_w - pad_left >= old_width) {
            output[pos] = pad_value_;
        } else {
            output[pos] = input[(block_num * old_height + padded_h - pad_top) * old_width + padded_w - pad_left];
        }
    }
    return;
}


void CallPad(const size_t size, const void* input, const int num, const int channels, const int old_height,
    const int old_width, const int padded_height, const int padded_width, const int pad_top, const int pad_left,
    float pad_value, void* output, MNN::CUDARuntime *runtime, MNN::DataType data_type)
{
    cudaStream_t cuda_stream = runtime->stream();
    int block_num = runtime->blocks_num(size);
    int threads_num = runtime->threads_num();
    cudaDataType_t cuda_type = kCudaDtypeMap[data_type];
    if(cuda_type == CUDA_R_32F)
        Pad<<<block_num, threads_num, 0, cuda_stream>>>(size, (float*)input, num, channels, old_height, old_width,
          padded_height, padded_width, pad_top, pad_left, pad_value, (float*)output);
    else if(cuda_type == CUDA_R_16F)
        Pad<<<block_num, threads_num, 0, cuda_stream>>>(size, (half*)input, num, channels, old_height, old_width,
          padded_height, padded_width, pad_top, pad_left, pad_value, (half*)output);
    else
        MNN_PRINT("current only support fp32 and fp16!!!!\n");
    return;
}