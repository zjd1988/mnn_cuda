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

#ifndef __SOFTMAX_IMPL_CUH__
#define __SOFTMAX_IMPL_CUH__

#include <vector>
#include <cuda_runtime.h>
#include "MNN_generated.h"
#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include "backend/cuda/execution/kernel_impl/kernel_constants.cuh"
#include "backend/cuda/execution/kernel_impl/transpose_impl.cuh"



class SoftmaxGpuKernel {
public:
    SoftmaxGpuKernel();
    ~SoftmaxGpuKernel();

    const std::vector<size_t> &GetInputSizeList() const;
    const std::vector<size_t> &GetOutputSizeList() const;
    const std::vector<size_t> &GetWorkspaceSizeList() const;

    bool Launch(std::vector<const void*> &inputs, std::vector<void*> &workspace, std::vector<void*> &outputs,
        MNN::CUDARuntime* runtime, MNN::DataType data_type);
    bool Init(const int axis, MNN::CUDARuntime* runtime, const std::vector<int> &input_shape,
    const std::vector<int> &output_shape, MNN::DataType data_type);

private:
    void InitResource(MNN::CUDARuntime *runtime);
    void InitSizeLists(); 
    void DestroyResource() noexcept;
    void InitSizeByAxis(const std::vector<int> &input_shape, const int &axis);
    void InitSizeByAxis2D(const std::vector<int> &input_shape, const int &axis);
    void InitSizeByAxisLastDim(const std::vector<int> &input_shape, const int &axis);

    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_descriptor_;
    cudnnTensorDescriptor_t output_descriptor_;
    cudnnSoftmaxAlgorithm_t algo_;
    cudnnSoftmaxMode_t mode_;
    cudnnDataType_t cudnn_data_type_;
    int cudnn_data_type_len_;
    size_t input_size_;
    size_t output_size_;
    size_t workspace_size_;
    std::vector<size_t> input_size_list_;
    std::vector<size_t> output_size_list_;
    std::vector<size_t> workspace_size_list_;

    std::vector<int> input_shape_;
    std::vector<int> transpose_shape_;
    std::vector<int> transpose_axis_;
    int axis_;
    int shape_size_;

    size_t batch_size_;
    size_t channel_size_;
    size_t height_;
    size_t width_;
};

#endif  // __SOFTMAX_IMPL_CUH__
