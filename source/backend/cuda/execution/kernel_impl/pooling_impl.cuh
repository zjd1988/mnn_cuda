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

#ifndef __POOLING_IMPL_CUH__
#define __POOLING_IMPL_CUH__

#include <vector>
#include <string>
#include <algorithm>
#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include "MNN_generated.h"
#include "backend/cuda/execution/kernel_impl/kernel_constants.cuh"
#include "backend/cuda/execution/kernel_impl/pad_impl.cuh"



class PoolingGpuFwdKernel {
public:
    PoolingGpuFwdKernel();
    ~PoolingGpuFwdKernel();

    const std::vector<size_t> &GetInputSizeList() const;
    const std::vector<size_t> &GetOutputSizeList() const;
    const std::vector<size_t> &GetWorkspaceSizeList() const;
    int GetPadSize();
    bool Launch(const std::vector<const void*> &inputs, const std::vector<void*> &workspace, 
        const std::vector<void*> &outputs, MNN::CUDARuntime* runtime, MNN::DataType data_type);
    bool Init(const MNN::Pool *node_para, MNN::CUDARuntime* runtime, std::vector<int> &input_shape, std::vector<int> &output_shape);

private:
    void InitResource(MNN::CUDARuntime *runtime);
    void InitSizeLists();
    void SetPad(const std::vector<int> &input_shape);
    void SetPoolingMode(const MNN::Pool *node_para);
    void DestroyResource() noexcept;

    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_descriptor_;
    cudnnTensorDescriptor_t output_descriptor_;
    cudnnPoolingDescriptor_t pooling_descriptor_;
    cudnnTensorDescriptor_t padded_descriptor_;
    cudnnPoolingMode_t pooling_mode_ = CUDNN_POOLING_MAX;
    MNN::PoolType mode_;
    MNN::PoolPadType pad_mode_;
    std::vector<size_t> input_size_list_;
    std::vector<size_t> output_size_list_;
    std::vector<size_t> workspace_size_list_;
    cudnnDataType_t cudnn_data_type_;
    int cudnn_data_type_len_;
    int old_height_;
    int old_width_;
    int pad_height_;
    int pad_width_;
    int pad_top_;
    int pad_left_;
    int stride_height_;
    int stride_width_;
    int window_height_;
    int window_width_;
    int n_;
    int c_;
    float pad_value_;
    size_t input_size_;
    size_t output_size_;
    size_t padded_size_;
    size_t workspace_size_;
    bool use_pad_;
    
};

#endif  // __POOLING_IMPL_CUH__
