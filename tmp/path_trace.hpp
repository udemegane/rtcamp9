#pragma once
#include <array>
#include <vulkan/vulkan_core.h>

#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "nvh/primitives.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/gltf_scene.hpp"
#include "nvvkhl/gltf_scene_vk.hpp"
#include "nvvkhl/gltf_scene_rtx.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_testing.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"

#include "shaders/dh_bindings.h"
#include "shaders/device_host.h"
#include "nvvkhl/shaders/dh_sky.h"

#include "nvvk/specialization.hpp"
#include "nvvk/images_vk.hpp"

#include "scene_resources.hpp"

namespace rtcamp9
{
    class PathTrace
    {
    public:
        PathTrace(nvvk::Context *ctx, nvvkhl::AllocVma *alloc);
        ~PathTrace();

        void createComputePipeline();
        VkCommandBuffer prepareResources(VkCommandBuffer cmd, const VkExtent2D &size);
        void updateComputeDescriptorSets();
        void updateComputeDescriptorSets(VkDescriptorImageInfo outImage);
        void runCompute(VkCommandBuffer cmd, const VkExtent2D &size);
        bool onUI();
        void setScene(const std::shared_ptr<SceneResources> &sceneResources);
        void resetFrame();
        // void prepareOutBuffer(VkCommandBuffer cmd, const VkExtent2D &size);
        VkDescriptorBufferInfo getOutBuffer();

    private:
        // void createGbuffers(const nvmath::vec2f &size);
        // void createVkBuffers();
        nvvk::RaytracingBuilderKHR::BlasInput primitiveToGeometry(const nvh::PrimitiveMesh &prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress);
        void createBottomLevelAS();
        void createTopLevelAS();
        void createComputePipelines();
        void pushDescriptorSet(VkCommandBuffer cmd);
        bool updateFrame();

        nvvk::Context *m_ctx{nullptr};
        std::unique_ptr<nvvk::DebugUtil> m_dutil;
        std::unique_ptr<nvvkhl::AllocVma> m_alloc;
        std::unique_ptr<nvvk::DescriptorSetContainer> m_rtSet;
        std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;
        std::shared_ptr<SceneResources> m_sceneResources{};
        std::unique_ptr<nvvkhl::Scene> m_scene;

        std::unique_ptr<VkDescriptorBufferInfo> m_outBuffer;
        std::unique_ptr<VkDescriptorImageInfo> m_oimage;
        std::vector<VkWriteDescriptorSet> m_writes;

        PushConstant m_pushConst{};
        FrameInfo m_finfo{};
        VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_graphicsPipeline = VK_NULL_HANDLE;
        int m_frame{0};
        int m_maxFrames{10000};

        VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
        nvvk::SBTWrapper m_sbt;
        nvvk::RaytracingBuilderKHR m_rtBuilder;
        nvvkhl::PipelineContainer m_rtPipe;
    };
}