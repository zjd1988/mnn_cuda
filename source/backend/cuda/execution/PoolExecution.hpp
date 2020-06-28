//
//  PoolExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef PoolExecution_hpp
#define PoolExecution_hpp

#include <array>
#include <memory>
#include <vector>
#include "core/Execution.hpp"
#include "backend/cuda/core/CUDABackend.hpp"
#include "backend/cuda/core/CUDARunningUtils.hpp"
#include "backend/cuda/execution/kernel_impl/pooling_impl.cuh"
namespace MNN {
namespace CUDA {

class PoolExecution : public Execution {
public:
    PoolExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~PoolExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Pool *mPoolParams;
    PoolType mPoolType;
    PoolPadType mPadType;
    DataType mDataType;
    CUDABackend *mCudaBackend;
    std::shared_ptr<PoolingGpuFwdKernel> mPoolCall;
    void* mWorkspace;
};

} // namespace CUDA
} // namespace MNN
#endif /* PoolExecution_hpp */
