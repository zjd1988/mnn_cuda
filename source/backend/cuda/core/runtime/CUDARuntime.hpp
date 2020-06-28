//
//  OpenCLRuntime.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpenCLRuntime_hpp
#define OpenCLRuntime_hpp


#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <sstream>
#include <string>
#include <vector>
#include "core/Macro.h"
#include "Type_generated.h"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cudnn.h>
#include <cuda.h>
#if CUDA_VERSION >= 10010
#include <cublasLt.h>
#endif

typedef enum {
    CUDA_FLOAT32 = 0,
    CUDA_FLOAT16 = 1,
} MNNCUDADataType_t;

typedef enum {
    MNNMemcpyHostToDevice = 1,
    MNNMemcpyDeviceToHost = 2,
    MNNMemcpyDeviceToDevice = 3,
} MNNMemcpyKind_t;

#define cuda_check(_x)                                       \
    do {                                                     \
        cudaError_t _err = (_x);                             \
        if (_err != cudaSuccess) {                           \
            MNN_CHECK(_err, #_x);                            \
        }                                                    \
    } while (0)

#define cublas_check(_x)                                       \
    do {                                                       \
        cublasStatus_t _err = (_x);                            \
        if (_err != CUBLAS_STATUS_SUCCESS) {                   \
            MNN_CHECK(_err, #_x);                              \
        }                                                      \
    } while (0)

#define cudnn_check(_x)                                       \
    do {                                                      \
        cudnnStatus_t _err = (_x);                            \
        if (_err != CUDNN_STATUS_SUCCESS) {                   \
            MNN_CHECK(_err, #_x);                             \
        }                                                     \
    } while (0)

#define cusolver_check(_x)                                       \
    do {                                                         \
        cusolverStatus_t _err = (_x);                            \
        if (_err != CUSOLVER_STATUS_SUCCESS) {                   \
            MNN_CHECK(_err, #_x);                                \
        }                                                        \
    } while (0)

#define after_kernel_launch()           \
    do {                                \
        cuda_check(cudaGetLastError()); \
    } while (0)




namespace MNN {

class CUDARuntime {
public:
    CUDARuntime(bool permitFloat16, int device_id);
    ~CUDARuntime();
    CUDARuntime(const CUDARuntime &) = delete;
    CUDARuntime &operator=(const CUDARuntime &) = delete;

    bool isSupportedFP16() const;
    bool isSupportedDotInt8() const;
    bool isSupportedDotAccInt8() const;

    std::vector<size_t> getMaxImage2DSize();
    bool isCreateError() const;

    float flops() const {
        return mFlops;
    }
    int device_id() const;
    size_t mem_alignment_in_bytes() const;
    void activate();
    void *malloc(size_t size_in_bytes);
    void free(void *ptr);

    void memcpy(void *dst, const void *src, size_t size_in_bytes, MNNMemcpyKind_t kind, bool sync = false);
    void memset(void *dst, int value, size_t size_in_bytes);
    void synchronize();
    cudaStream_t stream() const;
    cudnnHandle_t cudnn_handle();
    cublasHandle_t cublas_handle();
    #if CUDA_VERSION >= 10010
    cublasLtHandle_t cublasLt_handle();
    #endif
    void initialize_cusolver();
    cusolverDnHandle_t cusolver_handle();

    int threads_num() const { return mProp.maxThreadsPerBlock; }
    int major_sm() const { return mProp.major; }
    int blocks_num(const int total_threads) const {
        return std::min(((total_threads - 1) / mProp.maxThreadsPerBlock) + 1, mProp.multiProcessorCount);
    }

private:
    cudaDeviceProp mProp;
    int mDeviceId;
    cudaStream_t mStream = nullptr;

    cudnnHandle_t mCudnnHandle;
    cublasHandle_t mCublasHandle;
#if CUDA_VERSION >= 10010
    cublasLtHandle_t mCublasLtHandle;
#endif
    cusolverDnHandle_t mCusolverHandle;
    std::once_flag mCusolverInitialized;

    // struct ConstScalars {
    //     union FP16 {
    //         __half cuda_x;
    //         dt_float16 dnn_x;
    //         FP16() {}
    //     };
    //     static_assert(sizeof(FP16) == 2, "bad FP16 size");
    //     FP16 f16[2];
    //     dt_float32 f32[2];
    //     dt_int32 i32[2];
    //     void init();
    // };
    // //! device ptr to const scalars
    // ConstScalars* mConstScalars;

    bool mIsSupportedFP16     = false;
    bool mSupportDotInt8 = false;
    bool mSupportDotAccInt8 = false;
    float mFlops = 4.0f;
    bool mIsCreateError{false};

};

} // namespace MNN
#endif  /* CUDARuntime_hpp */
