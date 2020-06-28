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

 #include "backend/cuda/execution/kernel_impl/pooling_impl.cuh"

PoolingGpuFwdKernel::PoolingGpuFwdKernel()
{
    cudnn_handle_ = nullptr;
    input_descriptor_ = nullptr;
    output_descriptor_ = nullptr;
    pooling_descriptor_ = nullptr;
    padded_descriptor_ = nullptr;
    pooling_mode_ = CUDNN_POOLING_MAX;
    cudnn_data_type_ = CUDNN_DATA_FLOAT;
    old_height_ = 0;
    old_width_ = 0;
    pad_height_ = 0;
    pad_width_ = 0;
    pad_top_ = 0;
    pad_left_ = 0;
    stride_height_ = 0;
    stride_width_ = 0;
    window_height_ = 0;
    window_width_ = 0;
    n_ = 0;
    c_ = 0;
    pad_value_ = 0;
    input_size_ = 0;
    output_size_ = 0;
    padded_size_ = 0;
    workspace_size_ = 0;
    use_pad_ = true;
}


PoolingGpuFwdKernel::~PoolingGpuFwdKernel()
{
    DestroyResource();
}

const std::vector<size_t>& PoolingGpuFwdKernel::GetInputSizeList() const { return input_size_list_; }
const std::vector<size_t>& PoolingGpuFwdKernel::GetOutputSizeList() const { return output_size_list_; }
const std::vector<size_t>& PoolingGpuFwdKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool PoolingGpuFwdKernel::Launch(const std::vector<const void*> &inputs,const std::vector<void*> &workspace,
     const std::vector<void*> &outputs, MNN::CUDARuntime *runtime, MNN::DataType data_type) {

    const void *input_addr = inputs[0];
    void *output_addr = outputs[0];
    const float alpha = 1;
    const float beta = 0;
    // if (pad_mode_ == MNN::PoolPadType::PoolPadType_SAME && use_pad_) {
    //     void *padded_addr = workspace;
    //     CallPad(padded_size_ / cudnn_data_type_len_, input_addr, n_, c_, old_height_, old_width_, old_height_ + pad_height_,
    //         old_width_ + pad_width_, pad_top_, pad_left_, pad_value_, padded_addr, runtime, data_type);

    //     cudnn_check(cudnnPoolingForward(cudnn_handle_, pooling_descriptor_, &alpha, padded_descriptor_,
    //                                                 padded_addr, &beta, output_descriptor_, output_addr));
    // } else {
    cudnn_check(cudnnPoolingForward(cudnn_handle_, pooling_descriptor_, &alpha, input_descriptor_,
                                                input_addr, &beta, output_descriptor_, output_addr));
    // }
    return true;
}


bool PoolingGpuFwdKernel::Init(const MNN::Pool *node_para, MNN::CUDARuntime* runtime, 
    std::vector<int> &input_shape, std::vector<int> &output_shape) {
    InitResource(runtime);
    cudnn_data_type_ = kCudnnDtypeMap[node_para->dataType()];
    cudnn_data_type_len_ = kCudnnDtypeLenMap[node_para->dataType()];
    cudnn_check(cudnnSetTensor4dDescriptor(input_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, input_shape[0],
                                 input_shape[1], input_shape[2], input_shape[3]));
    cudnn_check(cudnnSetTensor4dDescriptor(output_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, output_shape[0],
                                 output_shape[1], output_shape[2], output_shape[3]));
    window_height_ = node_para->kernelY();
    window_width_ = node_para->kernelX();

    stride_height_ = node_para->strideY();
    stride_width_ = node_para->strideX();
    SetPoolingMode(node_para);
    // if (pad_mode_ == MNN::PoolPadType::PoolPadType_SAME) {
    //   SetPad(input_shape, window_height_, window_width_);
    // } else {
    SetPad(input_shape);
    cudnn_check(cudnnSetPooling2dDescriptor(pooling_descriptor_, pooling_mode_, CUDNN_NOT_PROPAGATE_NAN, window_height_,
                                    window_width_, pad_height_, pad_width_, stride_height_, stride_width_));
    // }

    InitSizeLists();
    return true;
}


void PoolingGpuFwdKernel::InitResource(MNN::CUDARuntime *runtime) {
    cudnn_handle_ = runtime->cudnn_handle();
    cudnn_check(cudnnCreateTensorDescriptor(&input_descriptor_));
    cudnn_check(cudnnCreateTensorDescriptor(&output_descriptor_));
    cudnn_check(cudnnCreateTensorDescriptor(&padded_descriptor_));
    cudnn_check(cudnnCreatePoolingDescriptor(&pooling_descriptor_));
    return;
}


void PoolingGpuFwdKernel::InitSizeLists() {

    cudnn_check(cudnnGetTensorSizeInBytes(input_descriptor_, reinterpret_cast<size_t *>(&input_size_)));
    cudnn_check(cudnnGetTensorSizeInBytes(output_descriptor_, reinterpret_cast<size_t *>(&output_size_)));

    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    if ((pad_mode_ == MNN::PoolPadType::PoolPadType_SAME) && use_pad_) {
        cudnn_check(cudnnGetTensorSizeInBytes(padded_descriptor_, reinterpret_cast<size_t *>(&padded_size_)));
        workspace_size_list_.push_back(padded_size_);
        if (padded_size_ == 0) {
            MNN_PRINT("Padded size is 0.");
            MNN_ASSERT(false);
        }
    }
    return;
}


void PoolingGpuFwdKernel::SetPad(const std::vector<int> &input_shape) {
    n_ = input_shape[0];
    c_ = input_shape[1];
    old_height_ = input_shape[2];
    old_width_ = input_shape[3];
    pad_height_ =
      std::max<int>(0, (((old_height_ / stride_height_) * stride_height_ == old_height_ ? (old_height_ / stride_height_)
                            : (old_height_ / stride_height_) + 1) - 1) * stride_height_ + window_height_ - old_height_);
    pad_width_ =
      std::max<int>(0, (((old_width_ / stride_width_) * stride_width_ == old_width_ ? (old_width_ / stride_width_)
                            : (old_width_ / stride_width_) + 1) - 1) * stride_width_ + window_width_ - old_width_);
    
    pad_top_ = pad_height_ / 2;
    pad_left_ = pad_width_ / 2;
    if (pad_height_ % 2 == 0 && pad_width_ % 2 == 0) {
      use_pad_ = false;
    }
    // cudnn_check(cudnnSetTensor4dDescriptor(padded_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, n_,
    //                 c_, old_height_ + pad_height_, old_width_ + pad_width_));
    cudnn_check(cudnnSetPooling2dDescriptor(pooling_descriptor_, pooling_mode_, CUDNN_NOT_PROPAGATE_NAN,
                    window_height_, window_width_, use_pad_ ? pad_top_ : 0,
                    use_pad_ ? pad_left_ : 0, stride_height_, stride_width_));
    return;
}


void PoolingGpuFwdKernel::SetPoolingMode(const MNN::Pool *node_para) {
    pad_mode_ = node_para->padType();
    mode_ = node_para->type();
    if (mode_ == MNN::PoolType::PoolType_AVEPOOL) {
      pooling_mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
      pad_value_ = 0.0;
    } else {
      pooling_mode_ = CUDNN_POOLING_MAX;
      pad_value_ = kSignedMinFloat;
    }
}

// int PoolingGpuFwdKernel::GetPadSize()
// {
//     return padded_size_;
// }


void PoolingGpuFwdKernel::DestroyResource() noexcept {
    cudnn_check(cudnnDestroyPoolingDescriptor(pooling_descriptor_));
    // cudnn_check(cudnnDestroyTensorDescriptor(padded_descriptor_));
    cudnn_check(cudnnDestroyTensorDescriptor(output_descriptor_));
    cudnn_check(cudnnDestroyTensorDescriptor(input_descriptor_));
}

