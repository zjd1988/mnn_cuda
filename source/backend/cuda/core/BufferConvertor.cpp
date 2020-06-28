//
//  ImageBufferConvertor.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cuda/core/BufferConvertor.hpp"

namespace MNN {
namespace CUDA {
bool convertFromNCHWBuffer(const Tensor *input, const Tensor *output, const MNN_DATA_FORMAT dstType,
                              CUDARuntime *runtime, bool needWait) {
    void *srcAddr = (void*)input->deviceId();
    void *dstAddr = (void*)output->deviceId();
    int size = input->size();
    if(dstType == MNN_DATA_FORMAT_NCHW)
        runtime->memcpy(dstAddr, srcAddr, size, MNNMemcpyDeviceToDevice, needWait);
    else
    {
        auto type = input->getType();
        auto inputShape = input->shape();
        int inputAxis[4] = {0, 3, 1, 2};
        if(type.code == halide_type_float && type.bits == 32)
            CallTranspose(size, srcAddr, &inputShape[0], &inputAxis[0], inputShape.size(), dstAddr, runtime, DataType_DT_FLOAT);
        else if(type.code == halide_type_float && type.bits == 16)
            CallTranspose(size, srcAddr, &inputShape[0], &inputAxis[0], inputShape.size(), dstAddr, runtime, DataType_DT_HALF);
        else
        {
            MNN_PRINT("current only support fp32 and fp16!!!!\n");
            return false;
        }
        if (true == needWait) {
            runtime->synchronize();
        }            
    }
    return true;
}

bool convertFromNHWCBuffer(const Tensor *input, const Tensor *output, const MNN_DATA_FORMAT dstType,
                              CUDARuntime *runtime, bool needWait) {
    void *srcAddr = (void*)input->deviceId();
    void *dstAddr = (void*)output->deviceId();
    int size = input->size();
    if(dstType == MNN_DATA_FORMAT_NHWC)
        runtime->memcpy(dstAddr, srcAddr, size, MNNMemcpyDeviceToDevice, needWait);
    else
    {
        auto type = input->getType();
        auto inputShape = input->shape();
        int inputAxis[4] = {0, 3, 1, 2};
        if(type.code == halide_type_float && type.bits == 32)
            CallTranspose(size, srcAddr, &inputShape[0], &inputAxis[0], inputShape.size(), dstAddr, runtime, DataType_DT_FLOAT);
        else if(type.code == halide_type_float && type.bits == 16)
            CallTranspose(size, srcAddr, &inputShape[0], &inputAxis[0], inputShape.size(), dstAddr, runtime, DataType_DT_HALF);
        else
        {
            MNN_PRINT("current only support fp32 and fp16!!!!\n");
            return false;
        }
        if (true == needWait) {
            runtime->synchronize();
        }
    }
    return true;
}

bool BufferConvertor::convertBuffer(const Tensor *srcBuffer, const MNN_DATA_FORMAT srcType, Tensor *dstBuffer,
                                            const MNN_DATA_FORMAT dstType, bool needWait) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start convertBuffer !\n");
#endif
    auto runtime = mCudaRuntime;
    switch (srcType) {
        case MNN_DATA_FORMAT_NHWC:
            convertFromNHWCBuffer(srcBuffer, dstBuffer, dstType, runtime, needWait);
            break;
        case MNN_DATA_FORMAT_NCHW:
            convertFromNCHWBuffer(srcBuffer, dstBuffer, dstType, runtime, needWait);
            break;
        default:
            MNN_PRINT("convertBuffer do nothing!\n");
            break;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end convertBuffer !\n");
#endif
    return true;
}

} // namespace CUDA
} // namespace MNN
