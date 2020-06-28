//
//  ConvExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvExecution_hpp
#define ConvExecution_hpp

#include <array>
#include <functional>
#include <memory>
#include <vector>
#include "core/Execution.hpp"
#include "backend/cuda/core/CUDABackend.hpp"
#include "backend/cuda/core/CUDARunningUtils.hpp"
#include "backend/cuda/execution/kernel_impl/conv2d_impl.cuh"
namespace MNN {
namespace CUDA {

class ConvExecution : public Execution {
public:
    ConvExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~ConvExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static std::shared_ptr<Tensor> getBias(CUDABackend *backend, const Convolution2D *conv);
private:
    const Convolution2DCommon *mConv2dCommonParams;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    std::shared_ptr<Tensor> mFilter;
    CUDABackend *mCudaBackend;
    std::shared_ptr<Conv2dGpuFwdKernel> mConv2DCall;
    std::shared_ptr<Tensor> mBias;
    std::shared_ptr<Tensor> mPad;
    std::shared_ptr<Tensor> mWorkspaceForward;

};

} // namespace CUDA
} // namespace MNN
#endif /* ConvExecution_hpp */
