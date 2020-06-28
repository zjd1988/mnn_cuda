//
//  SoftmaxExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SoftmaxExecution_hpp
#define SoftmaxExecution_hpp

#include <vector>
#include "core/Execution.hpp"
#include "backend/cuda/core/CUDABackend.hpp"
#include "backend/cuda/execution/kernel_impl/softmax_impl.cuh"
namespace MNN {
namespace CUDA {

class SoftmaxExecution : public Execution {
public:
    SoftmaxExecution(const std::vector<Tensor *> &inputs, int axis, Backend *backend);
    virtual ~SoftmaxExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    CUDABackend *mCUDABackend;
    std::shared_ptr<SoftmaxGpuKernel> mSoftmaxCall;
    std::vector<void*> mWorkspace;
    int mAxis;
};
} // namespace CUDA
} // namespace MNN
#endif /* SoftmaxExecution_hpp */
