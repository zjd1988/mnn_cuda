//
//  PoolExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cuda/execution/PoolExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

PoolExecution::PoolExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
    mCudaBackend = static_cast<CUDABackend *>(backend);
    mPoolParams    = op->main_as_Pool();
    mPoolType      = mPoolParams->type();
    mDataType      = mPoolParams->dataType();
    mPoolCall.reset(new PoolingGpuFwdKernel());
    mWorkspace = nullptr;
}

ErrorCode PoolExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start PoolExecution onResize !\n");
#endif

    auto input  = inputs[0];
    auto output = outputs[0];
    auto inputShape = input->shape();
    auto outputShape = output->shape();
    auto dataType = MNN::TensorUtils::getDataType(input);
    bool ret = mPoolCall.get()->Init(mPoolParams, mCudaBackend->getCUDARuntime(), inputShape, outputShape, dataType);
    // int bufferSize = mPoolCall.get()->GetPadSize();
    // auto bufferPool     = ((CUDABackend *)backend())->getBufferPool();

    // mWorkspace = bufferPool->alloc(bufferSize);
    // bufferPool->recycle(mWorkspace);
    // if(ret == false)
    // {
    //     MNN_PRINT("PoolExecution onResize error!!!\n");
    //     return COMPUTE_SIZE_ERROR;
    // }

#ifdef LOG_VERBOSE
    MNN_PRINT("end PoolExecution onResize !\n");
#endif
    return NO_ERROR;
}


ErrorCode PoolExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start PoolExecution onExecute !\n");
#endif

    const void* input = (const void*)inputs[0]->deviceId();
    void* output = (void*)inputs[0]->deviceId();
    std::vector<const void*> inputs_addr;
    std::vector<void*> outputs_addr;
    std::vector<void*> workspace;
    inputs_addr.push_back(input);
    outputs_addr.push_back(output);
    workspace.push_back(mWorkspace);
    mPoolCall.get()->Launch(inputs_addr, workspace, outputs_addr, mCudaBackend->getCUDARuntime(), mDataType);

#ifdef LOG_VERBOSE
    MNN_PRINT("end PoolExecution onExecute !\n");
#endif
    return NO_ERROR;
}

CUDACreatorRegister<TypedCreator<PoolExecution>> __PoolExecution(OpType_Pooling);
} // namespace CUDA
} // namespace MNN
