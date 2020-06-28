//
//  UnaryExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cuda/execution/UnaryExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

UnaryExecution::UnaryExecution(CudaUnaryOpType opType, Backend* backend) : Execution(backend) {
    auto cudaBackend = static_cast<CUDABackend*>(backend);
    auto mRuntime      = cudaBackend->getCUDARuntime();
    mOpType = opType;
}
ErrorCode UnaryExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto shape = inputs[0]->shape();
    mCount = 1;
    for(int i = 0; i < shape.size(); i++)
    {
        mCount*=shape[0];
    }
    return NO_ERROR;
}

ErrorCode UnaryExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start UnaryExecution onExecute...");
#endif
    auto type = inputs[0]->getType();
    if(type.code == halide_type_float && type.bits == 32)
        callUnary(inputs[0], outputs[0], mCount, mRuntime, CUDA_FLOAT32, mOpType);
    else if(type.code == halide_type_float && type.bits == 16)
        callUnary(inputs[0], outputs[0], mCount, mRuntime, CUDA_FLOAT16, mOpType);
    else
    {
        MNN_PRINT("Unary only support fp32 and fp16!!!!\n");
        return NOT_SUPPORT;
    }
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end UnaryExecution onExecute...");
#endif
    return NO_ERROR;
}

class UnaryCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (op->type() == OpType_UnaryOp) {
            switch (op->main_as_UnaryOp()->opType()) {
                case UnaryOpOperation_SQUARE:
                    return new UnaryExecution(CudaUnaryOpOperation_SQUARE, backend);
                case UnaryOpOperation_ERF:
                    return new UnaryExecution(CudaUnaryOpOperation_ERF, backend);
                case UnaryOpOperation_ERFC:
                    return new UnaryExecution(CudaUnaryOpOperation_ERFC, backend);
                case UnaryOpOperation_SQRT:
                    return new UnaryExecution(CudaUnaryOpOperation_SQRT, backend);
                case UnaryOpOperation_RSQRT:
                    return new UnaryExecution(CudaUnaryOpOperation_RSQRT, backend);
                case UnaryOpOperation_ABS:
                    return new UnaryExecution(CudaUnaryOpOperation_ABS, backend);
                case UnaryOpOperation_SIN:
                    return new UnaryExecution(CudaUnaryOpOperation_SIN, backend);
                case UnaryOpOperation_COS:
                    return new UnaryExecution(CudaUnaryOpOperation_COS, backend);
                case UnaryOpOperation_SIGN:
                    return new UnaryExecution(CudaUnaryOpOperation_SIGN, backend);
                case UnaryOpOperation_EXP:
                    return new UnaryExecution(CudaUnaryOpOperation_EXP, backend);
                case UnaryOpOperation_NEG:
                    return new UnaryExecution(CudaUnaryOpOperation_NEG, backend);
                case UnaryOpOperation_TAN:
                    return new UnaryExecution(CudaUnaryOpOperation_TAN, backend);
                case UnaryOpOperation_CEIL:
                    return new UnaryExecution(CudaUnaryOpOperation_CEIL, backend);
                case UnaryOpOperation_LOG1P:
                    return new UnaryExecution(CudaUnaryOpOperation_LOG1P, backend);                
                case UnaryOpOperation_FLOOR:
                    return new UnaryExecution(CudaUnaryOpOperation_FLOOR, backend);
                case UnaryOpOperation_ROUND:
                    return new UnaryExecution(CudaUnaryOpOperation_ROUND, backend);
                case UnaryOpOperation_RECIPROCAL:
                    return new UnaryExecution(CudaUnaryOpOperation_RECIPROCAL, backend);
                case UnaryOpOperation_LOG:
                    return new UnaryExecution(CudaUnaryOpOperation_LOG, backend);
                default:
                    break;
            }
            return nullptr;
        }
        if (op->type() == OpType_Sigmoid) {
            return new UnaryExecution(CudaUnaryOpOperation_SIGMOID, backend);
        }
        if (op->type() == OpType_TanH) {
            return new UnaryExecution(CudaUnaryOpOperation_TANH, backend);
        }
        return nullptr;
    }
};

CUDACreatorRegister<UnaryCreator> __UnaryExecution(OpType_UnaryOp);
CUDACreatorRegister<UnaryCreator> __SigmoidExecution(OpType_Sigmoid);
CUDACreatorRegister<UnaryCreator> __TanhExecution(OpType_TanH);
} // namespace CUDA
} // namespace MNN
