//
//  OpenCLBackend.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cuda/core/CUDABackend.hpp"
#include "MNN_generated.h"

#include "core/TensorUtils.hpp"
#include "core/SizeComputer.hpp"
#include <map>
#include <mutex>
#include <thread>
#include "core/Macro.h"
#include "backend/cuda/core/BufferConvertor.hpp"



namespace MNN {
namespace CUDA {

std::map<OpType, CUDABackend::Creator*>* gCreator() {
    static std::once_flag once;
    static std::map<OpType, CUDABackend::Creator*>* creators = nullptr;
    std::call_once(once, [&]() { creators = new std::map<OpType, CUDABackend::Creator*>; });
    return creators;
};

CUDABackend::CUDABackend(BackendConfig::PrecisionMode precision, BackendConfig::PowerMode power)
    : Backend(MNN_FORWARD_CUDA) {
    mPrecision = precision;
    // Shader precision
    if (precision == BackendConfig::Precision_Low) {
        mCUDARuntime.reset(new CUDARuntime(true, -1));
    } else {
        mCUDARuntime.reset(new CUDARuntime(false, -1));
    }
    if(mCUDARuntime.get()){
        if(mCUDARuntime->isCreateError() == true){
            mIsCreateError = true;
        }
        int ele_size = 1;
        if(precision == BackendConfig::Precision_Low)
        {
            ele_size = 2;
        }
        else
        {
            ele_size = 4;
        }
        // Mid memory precision
        mBufferPool.reset(new BufferPool(mCUDARuntime.get(), ele_size));
        mStaticBufferPool.reset(new BufferPool(mCUDARuntime.get(), ele_size));
        mBufferPoolInt8.reset(new BufferPoolInt8(mCUDARuntime.get(), sizeof(uint8_t)));
    }
}

CUDABackend::~CUDABackend() {
#ifdef LOG_VERBOSE
    MNN_PRINT("enter CUDABackend::~CUDABackend \n");
#endif
}

CUDARuntime* CUDABackend::getCUDARuntime() {
    return mCUDARuntime.get();
}

bool CUDABackend::onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start CUDABackend::onAcquireBuffer !\n");
#endif

    //int8
    if(nativeTensor->getType().code == halide_type_int && nativeTensor->getType().bits == 8){

        unsigned int size = nativeTensor->size();
#ifdef LOG_VERBOSE
    MNN_PRINT("enter int8 alloc ! size : %d \n", size);
#endif
        if (storageType == DYNAMIC_SEPERATE || storageType == STATIC) {
            auto buffer                               = mBufferPoolInt8->alloc(size, true);
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer; // fix
            return true;
        }
        if (storageType == DYNAMIC) {
            auto buffer                               = mBufferPoolInt8->alloc(size);
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer; // fix
            return true;
        }
        return false;
    }
    auto tensorShape = CUDA::tensorShapeFormat(nativeTensor);

    int N = tensorShape.at(0);
    int H = tensorShape.at(1);
    int W = tensorShape.at(2);
    int C = tensorShape.at(3);

    int mallocSize = N*H*W*C;
#ifdef LOG_VERBOSE
    MNN_PRINT("OpenCLBackend::onAcquireBuffer: [%d, %d, %d, %d], [%d, %d]\n", N, H, W, C, (int)imageWidth,
              (int)imageHeight);
#endif
    if (storageType == DYNAMIC_SEPERATE) {
        auto buffer                               = mBufferPool->alloc(mallocSize, true);
        ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer; // fix
        return true;
    }
    if (storageType == DYNAMIC) {
        auto buffer                               = mBufferPool->alloc(mallocSize, false);
        ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer; // fix
        return true;
    }
    MNN_ASSERT(storageType == STATIC);
    auto buffer                               = mStaticBufferPool->alloc(mallocSize, false);
    ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer; // fix
    return true;

//     size_t imageWidth  = (size_t)UP_DIV(C, 4) * W;
//     size_t imageHeight = (size_t)N * H;
// #ifdef LOG_VERBOSE
//     MNN_PRINT("OpenCLBackend::onAcquireBuffer: [%d, %d, %d, %d], [%d, %d]\n", N, H, W, C, (int)imageWidth,
//               (int)imageHeight);
// #endif
//     if (storageType == DYNAMIC_SEPERATE) {
//         auto buffer                               = mBufferPool->alloc(imageWidth, imageHeight, true);
//         ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer; // fix
//         return true;
//     }
//     if (storageType == DYNAMIC) {
//         auto buffer                               = mBufferPool->alloc(imageWidth, imageHeight, false);
//         ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer; // fix
//         return true;
//     }
//     MNN_ASSERT(storageType == STATIC);
//     auto buffer                               = mStaticBufferPool->alloc(imageWidth, imageHeight, false);
//     ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer; // fix
//     return true;
}

bool CUDABackend::onReleaseBuffer(const Tensor* nativeTensor, StorageType storageType) {
    if(nativeTensor->getType().code == halide_type_int && nativeTensor->getType().bits == 8){
        return true;
    }
    if (storageType == DYNAMIC_SEPERATE) {
        return true;
    }
    auto buffer = nativeTensor->deviceId();
    if (storageType == DYNAMIC) {
        mBufferPool->recycle((void*)buffer);
        return true;
    }
    if (storageType == STATIC) {
        mStaticBufferPool->recycle((void*)buffer, true);
    }
    return true;
}
bool CUDABackend::onAllocateBuffer() {
    return true;
}

bool CUDABackend::onClearBuffer() {
    mBufferPool->clear();
    mStaticBufferPool->clear();
    mBufferPoolInt8->clear();
    return true;
}
std::pair<float, bool> CUDABackend::onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) {
    auto creators = gCreator();
    auto iter      = creators->find(op->type());
    if (iter == creators->end()) {
        return std::make_pair(0.0f, false);
    }
    const float defaultScheduleTime = 0.05f;
    auto flops = SizeComputer::computeFlops(op, inputs, outputs);

    auto computeFlops = mCUDARuntime->flops();
    return std::make_pair(defaultScheduleTime + flops / 1024.0f / computeFlops * 1000.0f, true);
}
Execution* CUDABackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const MNN::Op* op) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start CUDABackend::onCreate \n");
#endif
    auto creators = gCreator();
    auto iter      = creators->find(op->type());

    if (iter == creators->end()) {
        if (nullptr != op->name()) {
            MNN_PRINT("Don't support type %s, %s\n", EnumNameOpType(op->type()), op->name()->c_str());
        } else {
            MNN_PRINT("Don't support type %s\n", EnumNameOpType(op->type()));
        }
        return NULL;
    }

    auto maxImageSize = mCUDARuntime->getMaxImage2DSize();
    bool valid        = true;
    for (auto t : inputs) {
        int imageHeight = t->batch() * t->height();
        int imageWidth  = t->width() * UP_DIV(t->channel(), 4);
        if (imageHeight > maxImageSize.at(0) || imageWidth > maxImageSize.at(1)) {
            valid = false;
            break;
        }
    }
    for (auto t : outputs) {
        int imageHeight = t->batch() * t->height();
        int imageWidth  = t->width() * UP_DIV(t->channel(), 4);
        if (imageHeight > maxImageSize.at(0) || imageWidth > maxImageSize.at(1)) {
            valid = false;
            break;
        }
    }
    if (!valid) {
        return NULL;
    }

    auto exe = iter->second->onCreate(inputs, outputs, op, this);
    if (NULL == exe) {
        if (nullptr != op->name()) {
            MNN_PRINT("The Creator Don't support type %d, %s\n", op->type(), op->name()->c_str());
        } else {
//            MNN_PRINT("The Creator Don't support type %s\n", EnumNameOpType(op->type()));
        }
        return NULL;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("End OpenCLBackend::onCreate \n");
#endif
    return exe;
}

void CUDABackend::onExecuteBegin() const {

}

void CUDABackend::onExecuteEnd() const {

}

bool CUDABackend::onWaitFinish() {
    mCUDARuntime.get()->synchronize();
    return true;
}

bool CUDABackend::isCreateError() const {
    return mIsCreateError;
}

void CUDABackend::_allocHostBuffer(int length) const {
    MNN_ASSERT(length > 0);
    if (nullptr != mHostBuffer.second && length <= mHostBuffer.first) {
        return;
    }
    mHostBuffer.first = length;
    auto ptr = mCUDARuntime->malloc(length);
    MNN_ASSERT(ptr != nullptr);
    mCUDARuntime->free(mHostBuffer.second);
    mHostBuffer.second = ptr;
}

void CUDABackend::copyFromDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const{
    auto needSize = dstTensor->size();
    auto hostPtr = dstTensor->host<float>();
    auto DeviceBuffer = (void*)srcTensor->deviceId();
    mCUDARuntime->memcpy(hostPtr, DeviceBuffer, needSize, MNNMemcpyDeviceToHost, true);
}

void CUDABackend::copyToDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const{
    auto needSize = srcTensor->size();
    auto hostPtr                = srcTensor->host<int8_t>();
    auto DeviceBuffer = (void*)dstTensor->deviceId();
    mCUDARuntime->memcpy(DeviceBuffer, hostPtr, needSize, MNNMemcpyHostToDevice);
}

void CUDABackend::copyFromDevice(const Tensor* srcTensor, const Tensor* dstTensor) const{
    std::vector<int> bufferShape = MNN::CUDA::tensorShapeFormat(srcTensor);
    MNN::Tensor interBuffer(0, Tensor::TENSORFLOW);
    interBuffer.buffer().dimensions = bufferShape.size();
    for (int i = 0; i < bufferShape.size(); i++) {
        interBuffer.buffer().dim[i].extent = bufferShape.at(i);
    }
    auto needSize = dstTensor->size();
    _allocHostBuffer(needSize);
    interBuffer.buffer().device = (uint64_t)mHostBuffer.second;

    MNN_DATA_FORMAT src_data_format = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
    MNN_DATA_FORMAT dst_data_format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
    switch (src_data_format) {
        case MNN_DATA_FORMAT_NHWC:
            CUDA::convertFromNHWCBuffer(srcTensor, &interBuffer, dst_data_format, mCUDARuntime.get());
            break;
        case MNN_DATA_FORMAT_NCHW:
            CUDA::convertFromNCHWBuffer(srcTensor, &interBuffer, dst_data_format, mCUDARuntime.get());
            break;
        default:
            MNN_PRINT("dont support this format!\n");
            break;
    }
    auto hostPtr = dstTensor->host<float>();

    mCUDARuntime->memcpy((void*)hostPtr, (void*)(interBuffer.buffer().device), needSize, MNNMemcpyDeviceToHost, true);
}
void CUDABackend::copyToDevice(const Tensor* srcTensor, const Tensor* dstTensor) const{
    std::vector<int> bufferShape = MNN::CUDA::tensorShapeFormat(srcTensor);
    MNN::Tensor interBuffer(0, Tensor::TENSORFLOW);
    interBuffer.buffer().dimensions = bufferShape.size();
    for (int i = 0; i < bufferShape.size(); i++) {
        interBuffer.buffer().dim[i].extent = bufferShape.at(i);
    }
    auto needSize = srcTensor->size();
    _allocHostBuffer(needSize);
    interBuffer.buffer().device = (uint64_t)mHostBuffer.second;
    auto hostPtr                = srcTensor->host<float>();
    mCUDARuntime->memcpy((void*)mHostBuffer.second, (void*)hostPtr, needSize, MNNMemcpyHostToDevice);
    // Host -> CUDA
    MNN_DATA_FORMAT src_data_format = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
    MNN_DATA_FORMAT dst_data_format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
    if (MNN_DATA_FORMAT_NHWC == src_data_format) {
        CUDA::convertFromNHWCBuffer(&interBuffer, dstTensor, dst_data_format, mCUDARuntime.get());
        return;
    }
    if (MNN_DATA_FORMAT_NCHW == src_data_format) {
        CUDA::convertFromNCHWBuffer(&interBuffer, dstTensor, dst_data_format, mCUDARuntime.get());
        return;
    }
    MNN_ASSERT(false);
    return;
}

void CUDABackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start onCopyBuffer !\n");
#endif
    //int8
    if(srcTensor->getType().code == halide_type_int && srcTensor->getType().bits == 8){
        if (srcTensor->deviceId() == 0 && dstTensor->deviceId() != 0) {
            copyToDeviceInt8(srcTensor, dstTensor);
        }else if(srcTensor->deviceId() != 0 && dstTensor->deviceId() == 0){
            copyFromDeviceInt8(srcTensor, dstTensor);
        }else{
            MNN_PRINT("onCopyBuffer int8 error !!! \n");
        }
    }else{
        if (srcTensor->deviceId() == 0 && dstTensor->deviceId() != 0) {
            copyToDevice(srcTensor, dstTensor);
        }else if(srcTensor->deviceId() != 0 && dstTensor->deviceId() == 0){
            copyFromDevice(srcTensor, dstTensor);
        }else{
            MNN_PRINT("onCopyBuffer float error !!! \n");
        }
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("end onCopyBuffer !\n");
#endif
}


bool CUDABackend::addCreator(OpType t, Creator* c) {
    auto map = gCreator();
    if (map->find(t) != map->end()) {
        MNN_PRINT("Error: %d type has be added\n", t);
        return false;
    }
    map->insert(std::make_pair(t, c));
    return true;
}

class CUDABackendCreator : public BackendCreator {
public:
    virtual Backend* onCreate(const Backend::Info& info) const override {
        BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
        BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
        if (nullptr != info.user) {
            precision = info.user->precision;
            power     = info.user->power;
        }
        auto backend = new CUDABackend(precision, power);
        if(backend != nullptr){
            if(!backend->isCreateError()){
                return backend;
            }else{
                delete backend;
            }
        }
        return nullptr;
    }
};

static const auto __cuda_global_initializer = []() {
    MNNInsertExtraBackendCreator(MNN_FORWARD_CUDA, new CUDABackendCreator, true);
    return true;
}();
} // namespace CUDA
} // namespace MNN
