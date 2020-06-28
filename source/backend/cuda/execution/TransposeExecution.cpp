//
//  UnaryExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cuda/execution/TransposeExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

TransposeExecution::TransposeExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
    auto cudaBackend = static_cast<CUDABackend*>(backend);
    auto mRuntime      = cudaBackend->getCUDARuntime();
}
ErrorCode TransposeExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const Tensor* input = inputs[0];
    const Tensor* perm  = inputs[1];
    mShape.clear();
    mPermutation.clear();
    auto shape = input->shape();
    mCount = 1;
    for(int i = 0; i < shape.size(); i++)
    {
        mCount*=shape[i];
        mShape.push_back(shape[i]);
    }

    for (int i = 0; i < perm->buffer().dim[0].extent; i++) {
        mPermutation.push_back(perm->host<int32_t>()[i]);
    }
    MNN_ASSERT(mShape.size() == mPermutation.size());
    return NO_ERROR;
}

ErrorCode TransposeExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start TransposeExecution onExecute...");
#endif
    auto type = inputs[0]->getType();
    if(type.code == halide_type_float && type.bits == 32)
        CallTranspose(mCount, inputs[0], &mShape[0], &mPermutation[0], mShape.size(), outputs[0], mRuntime, DataType_DT_FLOAT);
    else if(type.code == halide_type_float && type.bits == 16)
        CallTranspose(mCount, inputs[0], &mShape[0], &mPermutation[0], mShape.size(), outputs[0], mRuntime, DataType_DT_HALF);
    else
    {
        MNN_PRINT("transpose only support fp32 and fp16!!!!\n");
        return NOT_SUPPORT;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end TransposeExecution onExecute...");
#endif
    return NO_ERROR;
}

CUDACreatorRegister<TypedCreator<TransposeExecution>> __TransposeExecution(OpType_Transpose);
} // namespace CUDA
} // namespace MNN
