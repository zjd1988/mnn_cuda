//
//  BufferPool.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cuda/core/BufferPool.hpp"

namespace MNN {
namespace CUDA {

void* BufferPool::alloc(int size, bool seperate) {
    if (!seperate) {
        auto iter = mFreeList.lower_bound(size);
        if (iter != mFreeList.end()) {
            auto buffer = iter->second->buffer;
            mFreeList.erase(iter);
            return buffer;
        }
    }
    std::shared_ptr<BufferNode> node(new BufferNode(mCudaRuntime, size));
    mAllBuffer.insert(std::make_pair(node->buffer, node));
    return node->buffer;
}

void BufferPool::recycle(void* buffer, bool release) {
    auto iter = mAllBuffer.find(buffer);
    if (iter == mAllBuffer.end()) {
        MNN_ERROR("Error for recycle buffer\n");
        return;
    }
    if (release) {
        mAllBuffer.erase(iter);
        return;
    }
    mFreeList.insert(std::make_pair(iter->second->size, iter->second));
}

void BufferPool::clear() {
    mFreeList.clear();
    mAllBuffer.clear();
}

// void* BufferPool::alloc(int w, int h, bool seperate) {
//     int malloc_size = w * h * mElementSize * 4;
//     if (!seperate) {
//         int minWaste  = 0;
//         auto findIter = mFreeList.end();
//         for (auto iterP = mFreeList.begin(); iterP != mFreeList.end(); iterP++) {
//             auto& iter = *iterP;
//             if (iter->size >= malloc_size) {
//                 int waste = iter->size - malloc_size;
//                 if (minWaste == 0 || waste < minWaste) {
//                     findIter = iterP;
//                     minWaste = waste;
//                 }
//             }
//         }
//         if (findIter != mFreeList.end()) {
//             auto buffer = (*findIter)->buffer;
//             mFreeList.erase(findIter);
//             return buffer;
//         }
//     }
//     std::shared_ptr<BufferNode> node(new BufferNode(mCudaRuntime, malloc_size));
//     if (nullptr == node->buffer) {
//         MNN_ERROR("Alloc Image %d x %d error \n", w, h);
//         return nullptr;
//     }
//     mAllBuffer.insert(std::make_pair(node->buffer, node));
//     return node->buffer;
// }

void* BufferPoolInt8::alloc(int size, bool seperate) {
    if (!seperate) {
        auto iter = mFreeList.lower_bound(size);
        if (iter != mFreeList.end()) {
            auto buffer = iter->second->buffer;
            mFreeList.erase(iter);
            return buffer;
        }
    }
    std::shared_ptr<BufferNode> node(new BufferNode(mCudaRuntime, size));
    mAllBuffer.insert(std::make_pair(node->buffer, node));
    return node->buffer;
}

void BufferPoolInt8::recycle(void* buffer, bool release) {
    auto iter = mAllBuffer.find(buffer);
    if (iter == mAllBuffer.end()) {
        MNN_ERROR("Error for recycle buffer\n");
        return;
    }
    if (release) {
        mAllBuffer.erase(iter);
        return;
    }
    mFreeList.insert(std::make_pair(iter->second->size, iter->second));
}

void BufferPoolInt8::clear() {
    mFreeList.clear();
    mAllBuffer.clear();
}
} // namespace CUDA
} // namespace MNN
