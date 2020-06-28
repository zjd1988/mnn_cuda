//
//  ConvExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/cuda/execution/ConvExecution.hpp"

namespace MNN {
namespace CUDA {

ConvExecution::ConvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution init !\n");
#endif
    mCudaBackend                 = static_cast<CUDABackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mConv2dCommonParams            = conv2dCommonParams;
    mStrides                       = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mDilations                     = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};

    int biasSize             = conv2dParams->bias()->size();
    const float *biasDataPtr = conv2dParams->bias()->data();

    mConv2DCall.reset(new Conv2dGpuFwdKernel());
    

#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution init !\n");
#endif
}

ErrorCode ConvExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onResize !\n");
#endif
    auto input  = inputs[0];
    auto output = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    std::vector<int> filterShape = tensorShapeFormat(mFilter);
    std::vector<int> biasShape   = tensorShapeFormat(mBias);
    auto bias_data = (const void*)mBias->deviceId();
    auto bias_size = biasShape[0] * biasShape[1] * biasShape[2] * biasShape[3];
    auto dataType = MNN::TensorUtils::getDataType(input);
    mConv2DCall.get()->init( mConv2dCommonParams, mCudaBackend->getCUDARuntime(), inputShape, filterShape, outputShape, 
        bias_data, bias_size, dataType);
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onResize !\n");
#endif
    return NO_ERROR;
}

ErrorCode ConvExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start ConvExecution onExecute !\n");
#endif
    auto dataType = MNN::TensorUtils::getDataType(input);
    const void* input = (const void*)inputs[0]->deviceId();
    const void* filter = (const void*)mFilter.get()->deviceId();
    const void* bias = (const void*)mBias.get()->deviceId();
    void* output = (void*)inputs[0]->deviceId();
    void* pad = (void*)mPad.get()->deviceId();
    void* workspace_forward = (void *)mWorkspaceForward.get()->deviceId();
    std::vector<const void*> inputs_addr;
    std::vector<void*> outputs_addr;
    std::vector<void *> workspace;
    inputs_addr.push_back(input);
    inputs_addr.push_back(filter);
    inputs_addr.push_back(bias);
    outputs_addr.push_back(output);
    workspace.push_back(workspace_forward);
    if(mConv2DCall.get()->NeedPad())
        workspace.push_back(pad);
    mConv2DCall.get()->Launch(inputs, workspace, outputs, mCudaBackend->getCUDARuntime(), dataType);
#ifdef LOG_VERBOSE
    MNN_PRINT("end ConvExecution onExecute !\n");
#endif
    return NO_ERROR;
}

class ConvolutionCreator : public CUDABackend::Creator {
public:
    virtual ~ConvolutionCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new ConvExecution(inputs, op, backend);
    }
};

CUDACreatorRegister<ConvolutionCreator> __conv_op(OpType_Convolution);

} // namespace CUDA
} // namespace MNN
