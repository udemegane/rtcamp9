#pragma once

#include <vulkan/vulkan_core.h>
#include <vector>
#include <memory>

#include "nvvkhl/alloc_vma.hpp"
#include "nvvk/debug_util_vk.hpp"

#include "shaders/dh_reservoir.hlsl"
#include "shaders/device_host.h"

class DIReservoirContainer
{
public:
    DIReservoirContainer(nvvk::DebugUtil *dutil, nvvkhl::AllocVma *alloc) : m_dutil(dutil), m_alloc(alloc){};

    VkDescriptorBufferInfo getReservoir()
    {
        return m_reservoir;
    }

    VkDeviceSize getBufferSize()
    {
        return m_size;
    }

    VkCommandBuffer createReservoir(VkCommandBuffer cmd, const VkExtent2D &size)
    {
        DIReservoir data{};

        std::vector<DIReservoir>
            r_info(size.height * size.width, data);
        auto buffer = m_alloc->createBuffer(cmd, r_info, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        m_size = r_info.size();
        m_dutil->DBG_NAME(buffer.buffer);

        VkDescriptorBufferInfo bufInfo{buffer.buffer, 0, VK_WHOLE_SIZE};
        m_reservoir = bufInfo;
        return cmd;
    }

private:
    nvvk::DebugUtil *m_dutil;
    nvvkhl::AllocVma *m_alloc;
    VkDescriptorBufferInfo m_reservoir;
    VkDeviceSize m_size;
};

class ReservoirContainer
{
public:
    ReservoirContainer(nvvk::DebugUtil *dutil, nvvkhl::AllocVma *alloc) : m_dutil(dutil), m_alloc(alloc){};

    VkDescriptorBufferInfo getReservoir()
    {
        return m_reservoir;
    }

    VkDeviceSize getBufferSize()
    {
        return m_size;
    }

    VkCommandBuffer createReservoir(VkCommandBuffer cmd, const VkExtent2D &size)
    {
        PackedReservoir data{};

        std::vector<PackedReservoir>
            r_info(size.height * size.width * 1.5f, data);
        auto buffer = m_alloc->createBuffer(cmd, r_info, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        m_size = r_info.size();
        m_dutil->DBG_NAME(buffer.buffer);

        VkDescriptorBufferInfo bufInfo{buffer.buffer, 0, VK_WHOLE_SIZE};
        m_reservoir = bufInfo;
        return cmd;
    }

private:
    nvvk::DebugUtil *m_dutil;
    nvvkhl::AllocVma *m_alloc;
    VkDescriptorBufferInfo m_reservoir;
    VkDeviceSize m_size;
};
