//
//  SoftmaxExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cuda/execution/SoftmaxExecution.hpp"
#include "core/Macro.h"

namespace MNN {
namespace CUDA {

SoftmaxExecution::SoftmaxExecution(const std::vector<Tensor *> &inputs, int axis, Backend *backend)
    : Execution(backend) {
    mAxis          = axis;
    mCUDABackend = static_cast<CUDABackend *>(backend);
    mSoftmaxCall.reset(new SoftmaxGpuKernel());
}

ErrorCode SoftmaxExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];
    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    auto dataType = MNN::TensorUtils::getDataType(input);
    mSoftmaxCall.get()->Init(mAxis, mCUDABackend->getCUDARuntime(), inputShape, outputShape, dataType);

    auto bufferSizeList = mSoftmaxCall.get()->GetWorkspaceSizeList();
    auto bufferPool     = ((CUDABackend *)backend())->getBufferPool();
    for(int i = 0; i < bufferSizeList.size(); i++)
    {
        auto buffer = bufferPool->alloc(bufferSizeList[i]);
        if(buffer != nullptr) 
        {
            mWorkspace.push_back(buffer);
            bufferPool->recycle(buffer);
        }
        else
        {
            MNN_PRINT("out of memory");
            return OUT_OF_MEMORY;
        }
    }
    return NO_ERROR;
}

ErrorCode SoftmaxExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start SoftmaxExecution onExecute !\n");
#endif
    std::vector<const void*> inputAddr;
    std::vector<void*>       outputAddr;
    inputAddr.push_back((void*)inputs[0]->deviceId());
    outputAddr.push_back((void*)outputs[0]->deviceId());
    auto dataType = MNN::TensorUtils::getDataType(inputs[0]);
    mSoftmaxCall.get()->Launch(inputAddr, mWorkspace, outputAddr, mCUDABackend->getCUDARuntime(), dataType);
#ifdef LOG_VERBOSE
    MNN_PRINT("end SoftmaxExecution onExecute !\n");
#endif

    return NO_ERROR;
}

class SoftmaxCreator : public CUDABackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if(inputs[0]->dimensions() == 3 || outputs[0]->dimensions() == 3){
            MNN_PRINT("softmax not support dimensions == 3 \n");
            return nullptr;
        }
        auto dimType = inputs[0]->getDimensionType();
        if (dimType == Tensor::TENSORFLOW && inputs[0]->dimensions() == 4) {
            int index[4] = {0, 2, 3, 1};
            auto axis = op->main_as_Axis()->axis();
            if (axis < 0) {
                axis = inputs[0]->dimensions() + axis;
            }

            axis = index[axis];
            //1 : channel //0 : batch
            if (1 == axis || 0 == axis) {
                return new SoftmaxExecution(inputs, axis, backend);
            }
            return nullptr;
        } else {
            auto axis = op->main_as_Axis()->axis();
            if (axis < 0) {
                axis = inputs[0]->dimensions() + axis;
            }
            //1 : channel //0 : batch
            if (1 == axis || 0 == axis) {
                return new SoftmaxExecution(inputs, axis, backend);
            }
            return nullptr;
        }
    }
};
CUDACreatorRegister<SoftmaxCreator> __Softmax_op(OpType_Softmax);

} // namespace CUDA
} // namespace MNN
