//
//  ImageBufferConvertor.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef BufferConvertor_hpp
#define BufferConvertor_hpp

#include "core/Macro.h"
#include <MNN/Tensor.hpp>
#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include "backend/cuda/core/CUDARunningUtils.hpp"
#include "backend/cuda/execution/kernel_impl/transpose_impl.cuh"

namespace MNN {
namespace CUDA {
/**
 * @brief convert nchw buffer to image.
 * @param input      input tensor.
 * @param output     output tensor.
 * @param bufferToImageKernel    opencl kernel reference.
 * @param runtime    opencl runtime instance pointer.
 * @param needWait   whether need wait opencl complete before return or not, default false.
 * @return true if success, false otherwise.
 */
bool convertFromNCHWBuffer(const Tensor *input, const Tensor *output, const MNN_DATA_FORMAT dstType,
                              CUDARuntime *runtime, bool needWait = false);
/**
 * @brief convert from nhwc buffer to image.
 * @param input      input tensor.
 * @param output     output tensor.
 * @param bufferToImageKernel    opencl kernel reference.
 * @param runtime    opencl runtime instance pointer.
 * @param needWait   whether need wait opencl complete before return or not, default false.
 * @return true if success, false otherwise.
 */
bool convertFromNHWCBuffer(const Tensor *input, const Tensor *output, const MNN_DATA_FORMAT dstType,
                              CUDARuntime *runtime, bool needWait = false);


class BufferConvertor {
public:
    explicit BufferConvertor(CUDARuntime *cuda_runtime) : mCudaRuntime(cuda_runtime) {
    }

    bool convertBuffer(const Tensor *srcBuffer, const MNN_DATA_FORMAT srcType, Tensor *dstBuffer,
                        const MNN_DATA_FORMAT dstType, bool needWait = false);

private:
    CUDARuntime *mCudaRuntime;
};

} // namespace CUDA
} // namespace MNN
#endif  /* BufferConvertor_hpp */
