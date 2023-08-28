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
#include "shaders/dh_reservoir.hlsl"
#include "shaders/dh_reuse.h"
#include "shaders/device_host.h"

#include "nvvk/specialization.hpp"
#include "nvvk/images_vk.hpp"
#include "shader_compiler.hpp"
#include "resample_type.hpp"

class PathReuse
{
public:
    PathReuse(nvvk::Context *ctx, nvvkhl::AllocVma *alloc, HLSLShaderCompiler *compiler, EResampleType type);
    ~PathReuse();
    void createPipelineLayout();
    void createComputePipeline();
    void updateComputeDescriptorSets(VkDescriptorBufferInfo inReservoir, VkDescriptorBufferInfo outReservoir, VkDescriptorBufferInfo gbuffer, VkDescriptorBufferInfo sceneInfo);
    void updateConstants(const VkExtent2D &size);
    void runCompute(VkCommandBuffer cmd, const VkExtent2D &size);
    bool onUI();

private:
    HLSLShaderCompiler *m_compiler;
    nvvk::Context *m_ctx{nullptr};
    std::unique_ptr<nvvk::DebugUtil> m_dutil;

    std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;
    VkPipeline m_pipeline{VK_NULL_HANDLE};
    EResampleType m_type;
    GbufConst m_pushConst;
    std::unique_ptr<VkDescriptorBufferInfo> m_ireservoir;
    std::unique_ptr<VkDescriptorBufferInfo> m_gbuffer;
    std::unique_ptr<VkDescriptorBufferInfo> m_frameinfo;
    std::unique_ptr<VkDescriptorBufferInfo> m_oreservoir;
    std::vector<VkWriteDescriptorSet> m_writes;
};