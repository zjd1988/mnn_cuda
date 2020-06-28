//
//  OpenCLRuntime.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cuda/core/runtime/CUDARuntime.hpp"
#include <sys/stat.h>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "core/Macro.h"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define CUDNN_VERSION_STR STR(CUDNN_MAJOR) "." STR(CUDNN_MINOR) "." STR(CUDNN_PATCHLEVEL)

#pragma message "compile with cuda " STR(CUDART_VERSION) " "
#pragma message "compile with cuDNN " CUDNN_VERSION_STR " "

static_assert(!(CUDNN_MAJOR == 5 && CUDNN_MINOR == 1), "cuDNN 5.1.x series has bugs. Use 5.0.x instead.");

#undef STR
#undef STR_HELPER

namespace MNN {


bool CUDARuntime::isCreateError() const {
    return mIsCreateError;
}

// void CUDARuntime::ConstScalars::init() {
//     f16[0].dnn_x = 0; f16[1].dnn_x = 1;
//     f32[0] = 0; f32[1] = 1;
//     i32[0] = 0; i32[1] = 1;
// }

CUDARuntime::CUDARuntime(bool permitFloat16, int device_id) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start CUDARuntime !\n");
#endif
    int version;
    cuda_check(cudaRuntimeGetVersion(&version));
    if(version != CUDART_VERSION)
    {
        MNN_PRINT("megcore compiled with cuda %d, get %d at runtime\n", CUDART_VERSION, version);
        MNN_ASSERT(version == CUDART_VERSION);
    }
    int id = device_id;
    if (id < 0) {
        cuda_check(cudaGetDevice(&id));
    }
    mDeviceId = id;
    cuda_check(cudaGetDeviceProperties(&mProp, id));
    cuda_check(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));

#if CUDA_VERSION >= 10010
    if(cublasLtGetVersion() < 10010)
    {
        MNN_ASSERT(cublasLtGetVersion() >= 10010);
        MNN_PRINT("cuda library version is too low to run cublasLt");
    }
#endif
    cudnn_check(cudnnCreate(&mCudnnHandle));
    cublas_check(cublasCreate(&mCublasHandle));
#if CUDA_VERSION >= 10010
    cublas_check(cublasLtCreate(&mCublasLtHandle));
#endif    

    // Set stream for cuDNN and cublas handles.
    cudnn_check(cudnnSetStream(mCudnnHandle, mStream));
    cublas_check(cublasSetStream(mCublasHandle, mStream));

    // Note that all cublas scalars (alpha, beta) and scalar results such as dot
    // output resides at device side.
    cublas_check(cublasSetPointerMode(mCublasHandle, CUBLAS_POINTER_MODE_DEVICE));
    // init const scalars
    // cuda_check(cudaMalloc(&mConstScalars, sizeof(ConstScalars)));
    // ConstScalars const_scalars_val;
    // const_scalars_val.init();
    // cuda_check(cudaMemcpyAsync(m_const_scalars, &const_scalars_val,
    //             sizeof(ConstScalars), cudaMemcpyHostToDevice, mStream));
    // cuda_check(cudaStreamSynchronize(mStream));

}

CUDARuntime::~CUDARuntime() {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ~CUDARuntime !\n");
#endif
    if (mStream) {
        cuda_check(cudaStreamDestroy(mStream));
    }
    cudnn_check(cudnnDestroy(mCudnnHandle));
    cublas_check(cublasDestroy(mCublasHandle));
#if CUDA_VERSION >= 10010
    cublas_check(cublasLtDestroy(mCublasLtHandle));
#endif
    if (mCusolverHandle) {
        cusolver_check(cusolverDnDestroy(mCusolverHandle));
    }
    // cuda_check(cudaFree(mConstScalars));
#ifdef LOG_VERBOSE
    MNN_PRINT("end ~CUDARuntime !\n");
#endif
}

bool CUDARuntime::isSupportedFP16() const {
    return mIsSupportedFP16;
}

bool CUDARuntime::isSupportedDotInt8() const {
    return mSupportDotInt8;
}

bool CUDARuntime::isSupportedDotAccInt8() const {
    return mSupportDotAccInt8;
}

size_t CUDARuntime::mem_alignment_in_bytes() const {
    return std::max(mProp.textureAlignment, mProp.texturePitchAlignment);
}

int CUDARuntime::device_id() const {
    return mDeviceId;
}

void CUDARuntime::activate()
{
    int id = device_id();
    if (id >= 0) {
        cuda_check(cudaSetDevice(id));
    }
}

void *CUDARuntime::malloc(size_t size_in_bytes)
{
    void *ptr;
    cuda_check(cudaMalloc(&ptr, size_in_bytes));
    return ptr;
}

void CUDARuntime::free(void *ptr)
{
    cuda_check(cudaFree(ptr));
}

cudaStream_t CUDARuntime::stream() const {
    return mStream;
}

void CUDARuntime::memcpy(void *dst, const void *src,
        size_t size_in_bytes, MNNMemcpyKind_t kind, bool sync)
{
    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case MNNMemcpyDeviceToHost:
            cuda_kind = cudaMemcpyDeviceToHost;
            break;
        case MNNMemcpyHostToDevice:
            cuda_kind = cudaMemcpyHostToDevice;
            break;
        case MNNMemcpyDeviceToDevice:
            cuda_kind = cudaMemcpyDeviceToDevice;
            break;
        default:
            MNN_THROW("bad cuda memcpy kind\n");
    }
    if(sync == false)
        cuda_check(cudaMemcpyAsync(dst, src, size_in_bytes, cuda_kind, mStream));
    else
        cuda_check(cudaMemcpy(dst, src, size_in_bytes, cuda_kind));
}

void CUDARuntime::memset(void *dst, int value, size_t size_in_bytes)
{
    cuda_check(cudaMemsetAsync(dst, value, size_in_bytes, mStream));
}

void CUDARuntime::synchronize()
{
    cuda_check(cudaStreamSynchronize(mStream));
}

cudnnHandle_t CUDARuntime::cudnn_handle() {
    return mCudnnHandle;
}
cublasHandle_t CUDARuntime::cublas_handle() {
    return mCublasHandle;
}
#if CUDA_VERSION >= 10010
cublasLtHandle_t CUDARuntime::cublasLt_handle() {
    return mCublasLtHandle;
}
#endif
void CUDARuntime::initialize_cusolver() {
    cusolver_check(cusolverDnCreate(&mCusolverHandle));
    cusolver_check(cusolverDnSetStream(mCusolverHandle, mStream));
}
cusolverDnHandle_t CUDARuntime::cusolver_handle() {
    std::call_once(mCusolverInitialized,
                    [this] { initialize_cusolver(); });
    return mCusolverHandle;
}

} // namespace MNN
