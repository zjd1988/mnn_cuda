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

#ifndef __CONV2D_IMPL_CUH__
#define __CONV2D_IMPL_CUH__

#include <vector>
#include <string>
#include <algorithm>
#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include "MNN_generated.h"
#include "backend/cuda/execution/kernel_impl/kernel_constants.cuh"
#include "backend/cuda/execution/kernel_impl/pad_impl.cuh"

class Conv2dGpuFwdKernel{
public:
    Conv2dGpuFwdKernel();
    ~Conv2dGpuFwdKernel();
    const std::vector<size_t> &GetInputSizeList() const;
    const std::vector<size_t> &GetOutputSizeList() const;
    const std::vector<size_t> &GetWorkspaceSizeList() const;
    bool Launch(const std::vector<const void *> &inputs, const std::vector<void *> &workspace,
                const std::vector<void *> &outputs, MNN::CUDARuntime* runtime, MNN::DataType data_type);
    bool Init(const MNN::Convolution2DCommon *node_para, MNN::CUDARuntime* runtime, std::vector<int> &in_shape, 
      std::vector<int> &filter_shape, std::vector<int> &output_shape, const void* bias_data, const int bias_size, MNN::DataType data_type);
    
    bool NeedPad() {return use_pad_};

private:
    void InitResource(MNN::CUDARuntime *runtime);
    void InitSizeLists();
    void DestroyResource() noexcept;
    void SetPad(const std::vector<int> &in_shape, const MNN::Convolution2DCommon *kernel_node);
    void Set4DDesc(const std::vector<int> &in_shape, const std::vector<int> &filter_shape, const std::vector<int> &output_shape);
    void SelectAlgorithm(cudnnTensorDescriptor_t input_descriptor_real);
    void SetStrideAndDilation(const MNN::Convolution2DCommon *node_para);
    void SetActivationAndBias(const MNN::Convolution2DCommon *node_para, const void* bias_data, const int bias_size);
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    cudnnConvolutionFwdAlgo_t conv_algorithm_;
    cudnnConvolutionDescriptor_t conv_desc_;
    cudnnTensorDescriptor_t bias_desc_;
    cudnnTensorDescriptor_t padded_desc_;
    cudnnActivationDescriptor_t act_desc_;
    MNN::PadMode pad_mode_;
    std::vector<size_t> input_size_list_;
    std::vector<size_t> output_size_list_;
    std::vector<size_t> workspace_size_list_;
    const float pad_value_ = 0.0;
    cudnnDataType_t cudnn_data_type_;
    int cudnn_data_type_len_;
    int old_height_;
    int old_width_;
    int pad_height_;
    int pad_width_;
    int pad_top_;
    int pad_left_;
    int n_;
    int c_;
    int stride_x_;
    int stride_y_;
    int dilation_x_;
    int dilation_y_;
    int group_;
    size_t input_size_;
    size_t filter_size_;
    size_t output_size_;
    size_t padded_size_;
    size_t workspace_size_;
    bool use_pad_;
    bool use_relu_;
    bool use_relu6_;
    bool use_bias_;
};

#endif  // __CONV2D_IMPL_CUH__
