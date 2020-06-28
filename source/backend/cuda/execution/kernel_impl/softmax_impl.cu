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

#include "backend/cuda/execution/kernel_impl/softmax_impl.cuh"


SoftmaxGpuKernel::SoftmaxGpuKernel()
{
    cudnn_handle_ = nullptr;
    input_descriptor_ = nullptr;
    output_descriptor_ = nullptr;
    algo_ = CUDNN_SOFTMAX_ACCURATE;
    mode_ = CUDNN_SOFTMAX_MODE_INSTANCE;
    cudnn_data_type_ = CUDNN_DATA_FLOAT;
    cudnn_data_type_len_ = 0;
    input_size_ = 0;
    output_size_ = 0;
    workspace_size_ = 0;
    axis_ = 0;
    shape_size_ = 0;
    batch_size_ = 0;
    channel_size_ = 0;
    height_ = 0;
    width_ = 0;
}

SoftmaxGpuKernel::~SoftmaxGpuKernel()
{
    DestroyResource();
}
const std::vector<size_t>& SoftmaxGpuKernel::GetInputSizeList() const { return input_size_list_; }
const std::vector<size_t>& SoftmaxGpuKernel::GetOutputSizeList() const { return output_size_list_; }
const std::vector<size_t>& SoftmaxGpuKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool SoftmaxGpuKernel::Launch(std::vector<const void*> &inputs, std::vector<void*> &workspace, std::vector<void*> &outputs,
    MNN::CUDARuntime* runtime, MNN::DataType data_type) {

    const void*input_addr = inputs[0];
    void *output_addr = outputs[0];
    const float alpha = 1;
    const float beta = 0;
    if (axis_ == 1) {
        cudnn_check(cudnnSoftmaxForward(cudnn_handle_, algo_, mode_, &alpha, input_descriptor_,
                                                    input_addr, &beta, output_descriptor_, output_addr));
    } else {
        void *transpose_input_addr = workspace[0];
        void *transpose_output_addr = workspace[1];
        int *input_shape = (int*)workspace[2];
        int *transpose_shape = (int*)workspace[3];
        int *transpose_axis = (int*)workspace[4];
        cuda_check(cudaMemcpyAsync(input_shape, &input_shape_[0], workspace_size_, cudaMemcpyHostToDevice, runtime->stream()));
        cuda_check(cudaMemcpyAsync(transpose_shape, &transpose_shape_[0], workspace_size_, cudaMemcpyHostToDevice, runtime->stream()));
        cuda_check(cudaMemcpyAsync(transpose_axis, &transpose_axis_[0], workspace_size_, cudaMemcpyHostToDevice, runtime->stream()));
        int size = input_size_ / cudnn_data_type_len_;
        CallTranspose(size, input_addr, input_shape, transpose_axis, shape_size_, transpose_input_addr, runtime, data_type);
        cudnn_check(cudnnSoftmaxForward(cudnn_handle_, algo_, mode_, &alpha, input_descriptor_, transpose_input_addr, &beta,
                        output_descriptor_, transpose_output_addr));
        CallTranspose(size, transpose_output_addr, transpose_shape, transpose_axis, shape_size_, output_addr, runtime, data_type);
    }
    return true;
}

bool SoftmaxGpuKernel::Init(const int axis, MNN::CUDARuntime* runtime, 
    const std::vector<int> &input_shape, const std::vector<int> &output_shape, MNN::DataType data_type) {
    InitResource(runtime);
    cudnn_data_type_ = kCudnnDtypeMap[data_type];
    cudnn_data_type_len_ = kCudnnDtypeLenMap[data_type];
    shape_size_ = input_shape.size();
    algo_ = CUDNN_SOFTMAX_ACCURATE;
    InitSizeByAxis(input_shape, axis);

    cudnn_check(cudnnSetTensor4dDescriptor(input_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch_size_,
                        channel_size_, height_, width_));
    cudnn_check(cudnnSetTensor4dDescriptor(output_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch_size_,
                        channel_size_, height_, width_));
    InitSizeLists();
    return true;
}


void SoftmaxGpuKernel::InitResource(MNN::CUDARuntime *runtime) {
    cudnn_handle_ = runtime->cudnn_handle();
    cudnn_check(cudnnCreateTensorDescriptor(&input_descriptor_));
    cudnn_check(cudnnCreateTensorDescriptor(&output_descriptor_));
}

void SoftmaxGpuKernel::InitSizeLists() {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    workspace_size_list_.push_back(input_size_);
    workspace_size_list_.push_back(output_size_);
    workspace_size_list_.push_back(workspace_size_);
    workspace_size_list_.push_back(workspace_size_);
    workspace_size_list_.push_back(workspace_size_);
    return;
}


void SoftmaxGpuKernel::DestroyResource() noexcept {
    cudnn_check(cudnnDestroyTensorDescriptor(output_descriptor_));
    cudnn_check(cudnnDestroyTensorDescriptor(input_descriptor_));
}

void SoftmaxGpuKernel::InitSizeByAxis(const std::vector<int> &input_shape, const int &axis) {
    if (input_shape.size() == 2) {
        InitSizeByAxis2D(input_shape, axis);
    } else {
        InitSizeByAxisLastDim(input_shape, axis);
    }
}

void SoftmaxGpuKernel::InitSizeByAxis2D(const std::vector<int> &input_shape, const int &axis) {
    axis_ = axis;
    if (axis_ < 0) {
        axis_ += shape_size_;
    }
    if (axis_ == 1) {
        batch_size_ = input_shape[0];
        channel_size_ = input_shape[1];
    } else if (axis_ == 0) {
        batch_size_ = input_shape[1];
        channel_size_ = input_shape[0];
        input_shape_.push_back(input_shape[0]);
        input_shape_.push_back(input_shape[1]);
        transpose_shape_.push_back(input_shape[1]);
        transpose_shape_.push_back(input_shape[0]);
        transpose_axis_.push_back(1);
        transpose_axis_.push_back(0);
    } else {
        MNN_PRINT("Input is %d-D, but axis(%d) is invalid.", shape_size_, axis);
        MNN_ASSERT(false);        
    }

    height_ = 1;
    width_ = 1;
    input_size_ = cudnn_data_type_len_ * batch_size_ * channel_size_ * height_ * width_;
    output_size_ = input_size_;
    workspace_size_ = shape_size_ * sizeof(int);
}

void SoftmaxGpuKernel::InitSizeByAxisLastDim(const std::vector<int> &input_shape, const int &axis) {
    int axis_pos = axis;
    if (axis_pos < 0) {
        axis_pos += input_shape.size();
    }
    // axis should be -1 with ND
    if (axis_pos != input_shape.size() - 1) {
        MNN_PRINT("Input is %d-D, but axis(%d) is invalid.", shape_size_, axis);
        MNN_ASSERT(false);
    }
    // squeeze to 2d, then invoke cudnn
    size_t n = 1;
    for (size_t i = 0; i < input_shape.size() - 1; i++) {
        n *= input_shape[i];
    }
    axis_ = 1;
    batch_size_ = n;
    channel_size_ = input_shape[axis_pos];
    height_ = 1;
    width_ = 1;
    input_size_ = cudnn_data_type_len_ * batch_size_ * channel_size_ * height_ * width_;
    output_size_ = input_size_;
    input_shape_.push_back(batch_size_);
    input_shape_.push_back(channel_size_);
}