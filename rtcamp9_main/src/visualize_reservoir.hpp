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

#include "shaders/dh_bindings.h"
#include "shaders/device_host.h"
#include "shaders/dh_vis_binding.h"

#include "nvvk/specialization.hpp"
#include "nvvk/images_vk.hpp"
#include "shader_compiler.hpp"

class VisualizeReservoir
{
public:
    VisualizeReservoir(nvvk::Context *ctx, nvvkhl::AllocVma *alloc, HLSLShaderCompiler *compiler);
    ~VisualizeReservoir();

    void createPipelineLayout();
    void createComputePipeline();
    void updateComputeDescriptorSets(VkDescriptorBufferInfo inReservoir, VkDescriptorBufferInfo inGiReservoir, VkDescriptorImageInfo inThpImage, VkDescriptorImageInfo outImage);
    void updatePushConstant(PushConstant constant);
    void runCompute(VkCommandBuffer cmd, const VkExtent2D &size);

    bool onUI();

private:
    HLSLShaderCompiler *m_compiler;
    nvvk::Context *m_ctx{nullptr};
    std::unique_ptr<nvvk::DebugUtil> m_dutil;

    std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;
    VkPipeline m_pipeline{VK_NULL_HANDLE};

    PushConstant m_pushConst;
    std::unique_ptr<VkDescriptorBufferInfo> m_ibuffer;
    std::unique_ptr<VkDescriptorImageInfo> m_thpimage;
    std::unique_ptr<VkDescriptorBufferInfo> m_giReservoir;
    std::unique_ptr<VkDescriptorImageInfo> m_oimage;
    std::vector<VkWriteDescriptorSet> m_writes;
};