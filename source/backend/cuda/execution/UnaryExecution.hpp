//
//  UnaryExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef UnaryExecution_hpp
#define UnaryExecution_hpp

#include "core/Execution.hpp"

#include <vector>
#include "MNN_generated.h"
#include "backend/cuda/core/CUDABackend.hpp"
#include "backend/cuda/core/CUDARunningUtils.hpp"
#include "backend/cuda/execution/kernel_impl/unary_impl.cuh"

namespace MNN {
namespace CUDA {

class UnaryExecution : public Execution {
public:
    UnaryExecution(CudaUnaryOpType opType, Backend *backend);
    virtual ~UnaryExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    CUDARuntime *mRuntime;
    CudaUnaryOpType mOpType;
    int mCount;
};

} // namespace CUDA
} // namespace MNN
#endif /* UnaryExecution_hpp */
