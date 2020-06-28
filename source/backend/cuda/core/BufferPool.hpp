//
//  BufferPool.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef BufferPool_hpp
#define BufferPool_hpp

#include <map>
#include <memory>
#include <vector>
#include "core/NonCopyable.hpp"
#include "backend/cuda/core/runtime/CUDARuntime.hpp"

namespace MNN {
namespace CUDA {

class BufferPool : public NonCopyable {
public:
    BufferPool(CUDARuntime *runtime, int ele_size) {
        mCudaRuntime = runtime;
        mElementSize = ele_size;
    }

    void* alloc(int size, bool seperate = false);
    // void* alloc(int w, int h, bool seperate = false);
    void recycle(void* buffer, bool release = false);
    void clear();


    class BufferNode {
    public:
        int size;
        CUDARuntime *mCudaRuntime;
        void* buffer;
        BufferNode(CUDARuntime *runtime, int size) {
            mCudaRuntime = runtime;
            buffer = nullptr;
            this->size = 0;
            if(mCudaRuntime != nullptr) {
                buffer = mCudaRuntime->malloc(size);
                this->size = size;
            }
        }
        ~BufferNode() {
            if(buffer != nullptr) {
                mCudaRuntime->free(buffer);
                this->size = size;
            }

        }
    };
private:
    std::map<void*, std::shared_ptr<BufferNode>> mAllBuffer;
    std::multimap<int, std::shared_ptr<BufferNode>> mFreeList;
    CUDARuntime *mCudaRuntime;
    int mElementSize;
};
class BufferPoolInt8 : public NonCopyable {
public:
    BufferPoolInt8(CUDARuntime *runtime, int ele_size) {
        mCudaRuntime = runtime;
        mElementSize = ele_size;
    }

    void* alloc(int size, bool seperate = false);
    void recycle(void* buffer, bool release = false);
    void clear();

    class BufferNode {
    public:
        int size;
        CUDARuntime *mCudaRuntime;
        void* buffer;
        BufferNode(CUDARuntime *runtime, int size) {
            mCudaRuntime = runtime;
            buffer = nullptr;
            this->size = 0;
            if(mCudaRuntime != nullptr) {
                buffer = mCudaRuntime->malloc(size);
                this->size = size;
            }
        }
        ~BufferNode() {
            if(buffer != nullptr) {
                mCudaRuntime->free(buffer);
                this->size = 0;
            }
        }
    };
private:
    std::map<void*, std::shared_ptr<BufferNode>> mAllBuffer;
    std::multimap<int, std::shared_ptr<BufferNode>> mFreeList;
    CUDARuntime *mCudaRuntime;
    int mElementSize;
};
} // namespace CUDA
} // namespace MNN

#endif /* BufferPool_hpp */
