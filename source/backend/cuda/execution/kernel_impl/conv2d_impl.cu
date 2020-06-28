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
#include "backend/cuda/execution/kernel_impl/conv2d_impl.cuh"

static std::pair<int, int> convolutionPad(std::vector<int>& input_shape, std::vector<int>& output_shape, const MNN::Convolution2DCommon* mCommon) {
    if (mCommon->padMode() == MNN::PadMode_SAME) {
        int kernelWidthSize  = (mCommon->kernelX() - 1) * mCommon->dilateX() + 1;
        int kernelHeightSize = (mCommon->kernelY() - 1) * mCommon->dilateY() + 1;

        int padNeededWidth  = (output_shape[3] - 1) * mCommon->strideX() + kernelWidthSize - input_shape[3];
        int padNeededHeight = (output_shape[2] - 1) * mCommon->strideY() + kernelHeightSize - input_shape[2];
        auto mPadX               = padNeededWidth / 2;
        auto mPadY               = padNeededHeight / 2;
        return std::make_pair(mPadX, mPadY);
    }
    auto mPadX = mCommon->padX();
    auto mPadY = mCommon->padY();
    if (nullptr != mCommon->pads()) {
        mPadX = mCommon->pads()->data()[1];
        mPadY = mCommon->pads()->data()[0];
    }
    return std::make_pair(mPadX, mPadY);
}

Conv2dGpuFwdKernel::Conv2dGpuFwdKernel() {
    cudnn_handle_ = nullptr;
    input_desc_ = nullptr;
    output_desc_ = nullptr;
    filter_desc_ = nullptr;
    conv_desc_ = nullptr;
    padded_desc_ = nullptr;
    cudnn_data_type_ = CUDNN_DATA_FLOAT;
    cudnn_data_type_len_ = 0;
    old_height_ = 0;
    old_width_ = 0;
    pad_height_ = 0;
    pad_width_ = 0;
    pad_top_ = 0;
    pad_left_ = 0;
    n_ = 0;
    c_ = 0;
    group_ = 1;
    input_size_ = 0;
    filter_size_ = 0;
    output_size_ = 0;
    padded_size_ = 0;
    workspace_size_ = 0;
    use_pad_ = true;
    use_relu_ = false;
    use_relu6_ = false;
    use_bias_ = false;
}

Conv2dGpuFwdKernel::~Conv2dGpuFwdKernel() { DestroyResource(); }


bool Conv2dGpuFwdKernel::Launch(const std::vector<const void *> &inputs, const std::vector<void *> &workspace,
    const std::vector<void *> &outputs, MNN::CUDARuntime* runtime, MNN::DataType data_type) {

    const void *input_addr = inputs[0];
    const void *filter_addr = inputs[1];
    const void *bias_addr = inputs[2];
    void *output_addr = outputs[0];
    void *workspace_addr = nullptr;
    if (workspace_size_ != 0) {
        workspace_addr = workspace[0];
    }

    const float alpha = 1;
    const float beta = 0;
    if (pad_mode_ == MNN::PoolPadType::PoolPadType_SAME && use_pad_) {
        void *padded_addr = workspace[1];
        CallPad(padded_size_ / cudnn_data_type_len_, input_addr, n_, c_, old_height_, old_width_, old_height_ + pad_height_,
        old_width_ + pad_width_, pad_top_, pad_left_, pad_value_, padded_addr, runtime, data_type);
        if(use_bias_ && (use_relu_ || use_relu6_))
            cudnn_check(cudnnConvolutionBiasActivationForward(cudnn_handle_, &alpha, padded_desc_, padded_addr, filter_desc_, filter_addr, conv_desc_, 
                conv_algorithm_, workspace_addr_, workspace_size_, &beta, output_desc_, output_addr, bias_desc_, bias_addr, act_desc_, 
                output_desc_, output_addr));
        else
            cudnn_check(cudnnConvolutionForward(cudnn_handle_, &alpha, padded_desc_, padded_addr, filter_desc_, filter_addr, conv_desc_,
                conv_algorithm_, workspace_addr, workspace_size_, &beta, output_desc_, output_addr));
    } else {
        if(use_bias_ && (use_relu_ || use_relu6_))
            cudnn_check(cudnnConvolutionBiasActivationForward(cudnn_handle_, &alpha, input_desc_, input_addr, filter_desc_, filter_addr, conv_desc_, 
                conv_algorithm_, workspace_addr_, workspace_size_, &beta, output_desc_, output_addr, bias_desc_, bias_addr, act_desc_, 
                output_desc_, output_addr));
        else
            cudnn_check(cudnnConvolutionForward(cudnn_handle_, &alpha, input_desc_, input_addr, filter_desc_, filter_addr, conv_desc_,
                conv_algorithm_, workspace_addr, workspace_size_, &beta, output_desc_, output_addr));
    }
    if(use_bias_ == false && (use_relu_ || use_relu6_))
        cudnn_check(cudnnActivationForward(cudnn_handle_, act_desc_, &alpha, output_desc_, output_addr, &beta, output_desc_, output_addr));
    else if(use_bias_ && !(use_relu_ && use_relu6_))
        cudnn_check(cudnnAddTensor(cudnn_handle_, &alpha, bias_desc_, bias_addr, &alpha, output_desc_, output_addr));

    return true;
}
bool Conv2dGpuFwdKernel::Init(const MNN::Convolution2DCommon *node_para, MNN::CUDARuntime* runtime, std::vector<int> &in_shape, 
        std::vector<int> &filter_shape, std::vector<int> &output_shape, const void* bias_data, const int bias_size, MNN::DataType data_type) {
    InitResource(runtime);
    cudnn_data_type_ = kCudnnDtypeMap[data_type];
    cudnn_data_type_len_ = kCudnnDtypeLenMap[data_type];
    Set4DDesc(in_shape, filter_shape, output_shape);
    group_ = node_para->group();
    cudnn_check(cudnnSetConvolutionGroupCount(conv_desc_, group_));
    auto padxy = convolutionPad(in_shape, output_shape, node_para);
    pad_height_ = padxy.second;
    pad_width_ = padxy.first;
    pad_mode_ = node_para->padMode();
    SetStrideAndDilation(node_para);
    SetActivationAndBias(node_para, bias_data, bias_size);
    cudnnTensorDescriptor_t input_descriptor_real = nullptr;
    if (pad_mode_ == MNN::PadMode::PadMode_SAME) {
        SetPad(in_shape, node_para);
        input_descriptor_real = use_pad_ ? padded_desc_ : input_desc_;
    } else {
        if (pad_mode_ == MNN::PadMode::PadMode_VALID) {
            pad_height_ = 0;
            pad_width_ = 0;
        }
        cudnn_check(cudnnSetConvolution2dDescriptor(conv_desc_, pad_height_, pad_width_, stride_y_, stride_x_, dilation_y_,
                                    dilation_x_, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        input_descriptor_real = input_desc_;
    }
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
        cudnn_check(cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));
    }
    SelectAlgorithm(input_descriptor_real);
    InitSizeLists();
    return true;
}


void Conv2dGpuFwdKernel::InitResource(MNN::CUDARuntime *runtime) {
    cudnn_handle_ = runtime->cudnn_handle();
    cudnn_check(cudnnCreateTensorDescriptor(&input_desc_));
    cudnn_check(cudnnCreateTensorDescriptor(&output_desc_));
    cudnn_check(cudnnCreateTensorDescriptor(&padded_desc_));
    cudnn_check(cudnnCreateTensorDescriptor(&bias_desc_));
    cudnn_check(cudnnCreateFilterDescriptor(&filter_desc_));
    cudnn_check(cudnnCreateConvolutionDescriptor(&conv_desc_));
    cudnn_check(cudnnCreateActivationDescriptor(&act_desc_));
}

void Conv2dGpuFwdKernel::InitSizeLists() {
    cudnn_check(cudnnGetTensorSizeInBytes(input_desc_, reinterpret_cast<size_t *>(&input_size_)));
    cudnn_check(cudnnGetFilterSizeInBytes(filter_desc_, reinterpret_cast<size_t *>(&filter_size_)));
    cudnn_check(cudnnGetTensorSizeInBytes(output_desc_, reinterpret_cast<size_t *>(&output_size_)));
    cudnn_check(cudnnGetTensorSizeInBytes(padded_desc_, reinterpret_cast<size_t *>(&padded_size_)));
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(filter_size_);
    output_size_list_.push_back(output_size_);
    if (pad_mode_ == MNN::PoolPadType::PoolPadType_SAME && use_pad_) {
        cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_, padded_desc_, filter_desc_, conv_desc_, output_desc_,
                                            conv_algorithm_, &workspace_size_));
        workspace_size_list_.push_back(padded_size_);
    } else {
        cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_, input_desc_, filter_desc_, conv_desc_, output_desc_,
                                              conv_algorithm_, &workspace_size_));
    }
    (void)workspace_size_list_.insert(workspace_size_list_.begin(), workspace_size_);
    return;
}

void Conv2dGpuFwdKernel::DestroyResource() noexcept {
    cudnn_check(cudnnDestroyConvolutionDescriptor(conv_desc_));
    cudnn_check(cudnnDestroyFilterDescriptor(filter_desc_));
    cudnn_check(cudnnDestroyTensorDescriptor(padded_desc_));
    cudnn_check(cudnnDestroyTensorDescriptor(output_desc_));
    cudnn_check(cudnnDestroyTensorDescriptor(input_desc_));
    cudnn_check(cudnnDestroyTensorDescriptor(bias_desc_));
    cudnn_check(cudnnDestroyActivationDescriptor(act_desc_));
}

void Conv2dGpuFwdKernel::SetPad(const std::vector<int> &in_shape, const MNN::Convolution2DCommon *node_para) {
    n_ = in_shape[0];
    c_ = in_shape[1];
    old_height_ = in_shape[2];
    old_width_ = in_shape[3];
    pad_height_ = 2*node_para->padY();
    pad_width_ = 2*node_para->padX();
    pad_top_ = node_para->padY();
    pad_left_ = node_para->padX();

    // if use_pad_ == true, using zero padding in advance, else using the default cudnn pad.
    if (pad_height_ % 2 == 0 && pad_width_ % 2 == 0) {
        use_pad_ = false;
    }
    cudnn_check(cudnnSetTensor4dDescriptor(padded_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, n_, c_,
                                    old_height_ + pad_height_, old_width_ + pad_width_));
    cudnn_check(cudnnSetConvolution2dDescriptor(conv_desc_, use_pad_ ? 0 : pad_top_, use_pad_ ? 0 : pad_left_, stride_y_, stride_x_,
                                    dilation_y_, dilation_x_, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
}

void Conv2dGpuFwdKernel::Set4DDesc(const std::vector<int> &in_shape, const std::vector<int> &filter_shape,
               const std::vector<int> &output_shape) {
    cudnn_check(cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, in_shape[0],
                                in_shape[1], in_shape[2], in_shape[3]));

    cudnn_check(cudnnSetFilter4dDescriptor(filter_desc_, cudnn_data_type_, CUDNN_TENSOR_NCHW, filter_shape[0],
                                filter_shape[1], filter_shape[2], filter_shape[3]));
    cudnn_check(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, output_shape[0],
                                output_shape[1], output_shape[2], output_shape[3]));
}
void Conv2dGpuFwdKernel::SelectAlgorithm(cudnnTensorDescriptor_t input_descriptor_real) {
    if (group_ > 1 || CUDNN_MAJOR < 7) {
        cudnn_check(cudnnGetConvolutionForwardAlgorithm(cudnn_handle_, input_descriptor_real, filter_desc_, conv_desc_, output_desc_,
                                      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 0, &conv_algorithm_));
    } else {
        constexpr int requested_algo_count = 1;
        int returned_algo_count;
        cudnnConvolutionFwdAlgoPerf_t perf_results;
        cudnn_check(cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle_, input_descriptor_real, filter_desc_, conv_desc_,
                                                output_desc_, requested_algo_count, &returned_algo_count, &perf_results));
        conv_algorithm_ = perf_results.algo;
    }
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
        conv_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }
}
void Conv2dGpuFwdKernel::SetStrideAndDilation(const MNN::Convolution2DCommon *node_para) {
    stride_x_ = node_para->strideX();
    stride_y_ = node_para->strideY();
    dilation_x_ = node_para->dilateX();
    dilation_y_ = node_para->dilateY();
}

void Conv2dGpuFwdKernel::SetActivationAndBias(const MNN::Convolution2DCommon *node_para, const void* bias_data, const int bias_size)
{
    use_relu_ = node_para->relu();
    use_relu6_ = node_para->relu6();
    if(use_relu_)
        cudnn_check(cudnnSetActivationDescriptor(act_desc_, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
    else if(use_relu6_)
        cudnn_check(cudnnSetActivationDescriptor(act_desc_, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_NOT_PROPAGATE_NAN, 6.0));
    else
        MNN_PRINT("conv without activation!!!!!\n");
    
    if(bias_data)
    {
        int dim_bias[] = {1, bias_size, 1, 1};
        int stride_bias[] = {bias_size, 1, 1, 1};
        if(cudnn_data_type_ == CUDNN_DATA_FLOAT)
            cudnn_check(cudnnSetTensorNdDescriptor(bias_desc_, CUDNN_DATA_FLOAT, 4, dim_bias, stride_bias));
        else if(cudnn_data_type_ == CUDNN_DATA_HALF)
            cudnn_check(cudnnSetTensorNdDescriptor(bias_desc_, CUDNN_DATA_HALF, 4, dim_bias, stride_bias));
        else
            MNN_PRINT("only supports fp32/fp16 data type!!!\n");

        use_bias_ = true;
    }

}