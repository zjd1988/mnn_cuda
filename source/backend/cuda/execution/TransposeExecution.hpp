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
#include "backend/cuda/execution/kernel_impl/transpose_impl.cuh"

namespace MNN {
namespace CUDA {

class TransposeExecution : public Execution {
public:
    TransposeExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~TransposeExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    CUDARuntime *mRuntime;
    int mCount;
    std::vector<int32_t> mPermutation;
    std::vector<int32_t> mShape;
};

} // namespace CUDA
} // namespace MNN
#endif /* UnaryExecution_hpp */
