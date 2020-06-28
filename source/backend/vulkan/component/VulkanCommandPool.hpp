//
//  VulkanCommandPool.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanCommandPool_hpp
#define VulkanCommandPool_hpp

#include "core/NonCopyable.hpp"
#include "backend/vulkan/component/VulkanDevice.hpp"
#include "backend/vulkan/vulkan/vulkan_wrapper.h"
namespace MNN {
class VulkanCommandPool : public NonCopyable {
public:
    VulkanCommandPool(const VulkanDevice& dev);
    virtual ~VulkanCommandPool();

    class Buffer : public NonCopyable {
    public:
        Buffer(const VkCommandPool& pool, const VulkanDevice& dev);
        virtual ~Buffer();

        VkCommandBuffer get() const {
            return mBuffer;
        }

        void begin(VkCommandBufferUsageFlags flags) const;
        void end() const;
        void barrierSource(VkBuffer source, size_t start, size_t end) const;
        void barrierImage(VkImage source, VkImageLayout oldLayout, VkImageLayout newLayout) const;

    private:
        VkCommandBuffer mBuffer;
        const VkCommandPool mPool;
        const VulkanDevice& mDevice;
    };

    VulkanCommandPool::Buffer* allocBuffer() const;

    VkCommandPool pool() const {
        return mPool;
    }

    void submitAndWait(VkCommandBuffer buffer) const;

private:
    const VulkanDevice& mDevice;
    VkCommandPool mPool;
};
} // namespace MNN
#endif /* VulkanCommandPool_hpp */
