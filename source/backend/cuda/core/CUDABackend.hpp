//
//  OpenCLBackend.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CUDABackend_hpp
#define CUDABackend_hpp

#include "core/Backend.hpp"
#include "MNN_generated.h"
#include <list>
#include <vector>
#include "backend/cuda/core/BufferPool.hpp"
#include "core/Macro.h"
#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include "backend/cuda/core/CUDARunningUtils.hpp"

namespace MNN {
namespace CUDA {

class CUDABackend final : public Backend {
public:
    CUDABackend(BackendConfig::PrecisionMode precision, BackendConfig::PowerMode power);
    ~CUDABackend();

    CUDARuntime *getCUDARuntime();
    virtual bool onAcquireBuffer(const Tensor *nativeTensor, StorageType storageType) override;
    virtual bool onReleaseBuffer(const Tensor *nativeTensor, StorageType storageType) override;
    virtual bool onAllocateBuffer() override;
    virtual bool onClearBuffer() override;

    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op) override;

    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;

    virtual bool onWaitFinish() override;

    virtual void onCopyBuffer(const Tensor *srcTensor, const Tensor *dstTensor) const override;

    class Creator {
    public:
        virtual ~Creator() = default;
        virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &output, const MNN::Op *op, Backend *backend) const = 0;
    };

    static bool addCreator(OpType t, Creator *c);

    BufferPool *getBufferPool() const {
        return mBufferPool.get();
    }
    BackendConfig::PrecisionMode getPrecision() const {
        return mPrecision;
    }
    virtual std::pair<float, bool> onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                             const MNN::Op* op) override;

    bool isCreateError() const;

private:
    void copyFromDevice(const Tensor* srcTensor, const Tensor* dstTensor) const;
    void copyToDevice(const Tensor* srcTensor, const Tensor* dstTensor) const;
    void copyFromDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const;
    void copyToDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const;

    void _allocHostBuffer(int length) const;
    std::shared_ptr<BufferPool> mBufferPool;
    std::shared_ptr<BufferPool> mStaticBufferPool;
    std::shared_ptr<BufferPoolInt8> mBufferPoolInt8;
    std::shared_ptr<CUDARuntime> mCUDARuntime;

    mutable std::pair<int, void *> mHostBuffer;
    BackendConfig::PrecisionMode mPrecision;
    bool mIsCreateError{false};
};

template <class T>
class CUDACreatorRegister {
public:
    CUDACreatorRegister(OpType type) {
        T *t = new T;
        CUDABackend::addCreator(type, t);
    }
    ~CUDACreatorRegister() = default;
};

template <typename T>
class TypedCreator : public CUDABackend::Creator {
public:
    virtual ~TypedCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op,
                                Backend *backend) const override {
        return new T(inputs, op, backend);
    }
};

} // namespace CUDA
} // namespace MNN
#endif  /* CUDABackend_hpp */
