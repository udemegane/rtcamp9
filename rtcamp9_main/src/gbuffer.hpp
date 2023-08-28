#pragma once
#include <array>
#include <vulkan/vulkan_core.h>
#include "imgui/imgui_helper.h"
#include "nvh/primitives.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/pipeline_container.hpp"

#include "shaders/dh_gbuf.h"
#include "shaders/device_host.h"

#include "nvvk/specialization.hpp"
#include "nvvk/images_vk.hpp"
#include "shader_compiler.hpp"

class GBuffer
{
public:
    GBuffer(nvvk::Context *ctx, nvvkhl::AllocVma *alloc, HLSLShaderCompiler *compiler);
    ~GBuffer();
    void createPipelineLayout();
    void createComputePIpeline();
    void updateComputeDescriptorSets(VkWriteDescriptorSetAccelerationStructureKHR asInfo, VkDescriptorBufferInfo outBuffer, VkDescriptorBufferInfo frameInfo, VkDescriptorBufferInfo sceneInfo);
    void runCompute(VkCommandBuffer cmd, const VkExtent2D &size);
    bool onUI();

private:
    HLSLShaderCompiler *m_compiler;
    nvvk::Context *m_ctx{nullptr};
    std::unique_ptr<nvvk::DebugUtil> m_dutil;

    std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;
    VkPipeline m_pipeline{VK_NULL_HANDLE};

    GbufConst m_pushConst;
    std::unique_ptr<VkWriteDescriptorSetAccelerationStructureKHR> m_asinfo;
    std::unique_ptr<VkDescriptorBufferInfo> m_outbuffer;
    std::unique_ptr<VkDescriptorBufferInfo> m_frameinfo;
    std::unique_ptr<VkDescriptorBufferInfo> m_sceneinfo;
    std::vector<VkWriteDescriptorSet> m_writes;
};

class GBufferContainer
{
public:
    GBufferContainer(nvvk::DebugUtil *dutil, nvvkhl::AllocVma *alloc) : m_dutil(dutil), m_alloc(alloc){};

    VkDescriptorBufferInfo getGBuffer()
    {
        return m_reservoir;
    }

    VkDeviceSize getBufferSize()
    {
        return m_size;
    }

    VkCommandBuffer createGBuffer(VkCommandBuffer cmd, const VkExtent2D &size)
    {
        GBufStruct data{};

        std::vector<GBufStruct>
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