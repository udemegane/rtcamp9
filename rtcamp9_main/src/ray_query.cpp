/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

//////////////////////////////////////////////////////////////////////////
/*

    This shows the use of Ray Query, or casting rays in a compute shader

*/
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <queue>
#include <thread>
#include <string>
#include <filesystem>
#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

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
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_testing.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"
#include "visualize_reservoir.hpp"

#include "shaders/dh_bindings.h"
#include "shaders/device_host.h"
#include "shaders/dh_gbuf.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "shader_compiler.hpp"
#include "reservoirs.hpp"
#include "gbuffer.hpp"
#include "path_reuse.hpp"

#if USE_HLSL
#include "_autogen/ray_query_computeMain.spirv.h"
const auto &comp_shd = std::vector<char>{std::begin(ray_query_computeMain), std::end(ray_query_computeMain)};
#elif USE_SLANG
#include "_autogen/ray_query_computeMain.spirv.h"
const auto &comp_shd = std::vector<uint32_t>{std::begin(ray_query_computeMain), std::end(ray_query_computeMain)};
#else
#include "_autogen/ray_query.comp.h"
const auto &comp_shd = std::vector<uint32_t>{std::begin(ray_query_comp), std::end(ray_query_comp)};
#endif
#include "nvvk/specialization.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/buffers_vk.hpp"
#include "stb_image_write.h"

#define GROUP_SIZE 16 // Same group size as in compute shader

/// </summary> Ray trace multiple primitives using Ray Query
class RayQuery : public nvvkhl::IAppElement
{
  enum
  {
    eImgTonemapped,
    eImgRendered,
    eImgIntermediate1
  };

public:
  RayQuery(bool auto_render, uint spp) : m_auto_render(auto_render)
  {
    m_pushConst.maxSamples = spp;
  };
  ~RayQuery() override = default;

  void onAttach(nvvkhl::Application *app) override
  {
    if (m_auto_render)
      m_fullscreen = true;
    m_app = app;

    // m_app->
    m_device = m_app->getDevice();

    m_compiler = std::make_unique<HLSLShaderCompiler>();

    m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);                   // Debug utility
    m_alloc = std::make_unique<nvvkhl::AllocVma>(m_app->getContext().get()); // Allocator
    m_rtSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    // RenderPasses
    m_tonemapper = std::make_unique<nvvkhl::TonemapperPostProcess>(m_app->getContext().get(), m_alloc.get());
    m_resVisualizer = std::make_unique<VisualizeReservoir>(m_app->getContext().get(), m_alloc.get(), m_compiler.get());
    m_gbufferPass = std::make_unique<GBuffer>(m_app->getContext().get(), m_alloc.get(), m_compiler.get());
    m_spatialPathReusePass = std::make_unique<PathReuse>(m_app->getContext().get(), m_alloc.get(), m_compiler.get(), EResampleType::Spatial);
    m_temporalPathReusePass = std::make_unique<PathReuse>(m_app->getContext().get(), m_alloc.get(), m_compiler.get(), EResampleType::Temporal);

    // prepare structured buffers
    m_diResContainer = std::make_unique<DIReservoirContainer>(m_dutil.get(), m_alloc.get());
    m_initReservoirContainer = std::make_unique<ReservoirContainer>(m_dutil.get(), m_alloc.get());
    m_spatialReservoirContainer = std::make_unique<ReservoirContainer>(m_dutil.get(), m_alloc.get());
    m_temporalReservoirContainer = std::make_unique<ReservoirContainer>(m_dutil.get(), m_alloc.get());
    m_gbufferContainer = std::make_unique<GBufferContainer>(m_dutil.get(), m_alloc.get());

    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    int32_t gctQueueIndex = m_app->getContext()->m_queueGCT.familyIndex;
    m_rtBuilder.setup(m_device, m_alloc.get(), gctQueueIndex);
    m_sbt.setup(m_device, gctQueueIndex, m_alloc.get(), m_rtProperties);

    m_frameinfo = FrameInfo{};
    m_xpos = 0.0f;

    // Create resources
    createScene();
    createVkBuffers();
    createBottomLevelAS();
    createTopLevelAS();
#if USE_RTX
    createRtxPipeline();
#else
    createCompPipelines();
#endif

    m_tonemapper->createComputePipeline();
    m_resVisualizer->createPipelineLayout();
    m_resVisualizer->createComputePipeline();
    m_gbufferPass->createPipelineLayout();
    m_gbufferPass->createComputePIpeline();
    m_spatialPathReusePass->createPipelineLayout();
    m_spatialPathReusePass->createComputePipeline();
    m_temporalPathReusePass->createPipelineLayout();
    m_temporalPathReusePass->createComputePipeline();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void updatePassDescriptors()
  {
    VkAccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();
    VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    descASInfo.accelerationStructureCount = 1;
    descASInfo.pAccelerationStructures = &tlas;
    VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo sceneDesc{m_bSceneDesc.buffer, 0, VK_WHOLE_SIZE};

    m_gbufferPass->updateComputeDescriptorSets(
        m_rtBuilder.getAccelerationStructure(),
        m_gbufferContainer->getGBuffer(),
        dbi_unif,
        sceneDesc);
    m_temporalPathReusePass->updateComputeDescriptorSets(
        m_initReservoirContainer->getReservoir(),
        m_temporalReservoirContainer->getReservoir(),
        m_spatialReservoirContainer->getReservoir(),
        m_gbufferContainer->getGBuffer(),
        m_gBuffers->getDescriptorImageInfo(eImgIntermediate1),
        dbi_unif,
        sceneDesc);
    m_spatialPathReusePass->updateComputeDescriptorSets(
        m_spatialReservoirContainer->getReservoir(),
        m_initReservoirContainer->getReservoir(),
        m_gbufferContainer->getGBuffer(),
        m_gBuffers->getDescriptorImageInfo(eImgIntermediate1),
        dbi_unif,
        sceneDesc);

    m_resVisualizer->updateComputeDescriptorSets(
        m_diResContainer->getReservoir(),
        m_initReservoirContainer->getReservoir(),
        m_gBuffers->getDescriptorImageInfo(eImgIntermediate1),
        m_gBuffers->getDescriptorImageInfo(eImgRendered));

    m_tonemapper->updateComputeDescriptorSets(
        m_gBuffers->getDescriptorImageInfo(eImgRendered),
        m_gBuffers->getDescriptorImageInfo(eImgTonemapped));
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    auto cmd = m_app->createTempCmdBuffer();
    auto size = VkExtent2D{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    createGbuffers({width, height});
    m_gbufferPass->updatePushConstants(size);

    // recreate structuredbuffers
    {
      cmd = m_diResContainer->createReservoir(cmd, size);
      cmd = m_initReservoirContainer->createReservoir(cmd, size);
      cmd = m_temporalReservoirContainer->createReservoir(cmd, size);
      cmd = m_spatialReservoirContainer->createReservoir(cmd, size);
      cmd = m_gbufferContainer->createGBuffer(cmd, size);
    }

    // update descriptorsets
    updatePassDescriptors();

    // VkAccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();
    // VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    // descASInfo.accelerationStructureCount = 1;
    // descASInfo.pAccelerationStructures = &tlas;
    // VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    // VkDescriptorBufferInfo sceneDesc{m_bSceneDesc.buffer, 0, VK_WHOLE_SIZE};

    // m_gbufferPass->updateComputeDescriptorSets(descASInfo, m_gbufferContainer->getGBuffer(), dbi_unif, sceneDesc);
    // m_resVisualizer->updateComputeDescriptorSets(m_diResContainer->getReservoir(),
    //                                              m_gBuffers->getDescriptorImageInfo(eImgRendered));
    // m_tonemapper->updateComputeDescriptorSets(m_gBuffers->getDescriptorImageInfo(eImgRendered),
    //                                           m_gBuffers->getDescriptorImageInfo(eImgTonemapped));

    resetFrame();
    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  void onUIRender() override
  {
    if (!m_fullscreen)
    { // Setting menu
      ImGui::Begin("Settings");

      ImGuiH::CameraWidget();

      using namespace ImGuiH;

      if (ImGui::CollapsingHeader("Compiler", ImGuiTreeNodeFlags_DefaultOpen))
      {
        if (ImGui::Button("Reload Shaders"))
        {
          m_shaderCompile = true;
        }
        if (ImGui::Button("Close App"))
        {
          m_app->close();
        }
      }

      if (ImGui::CollapsingHeader("RTCamp9", ImGuiTreeNodeFlags_DefaultOpen))
      {
        PropertyEditor::begin();
        bool changed = PropertyEditor::entry("X Pos", [&]
                                             { return ImGui::SliderFloat("#1", &m_xpos, -50.0F, 100.0F); });
        if (changed)
        {
          CameraManip.setLookat({-15.0F + m_xpos, 4.33F, 0.0f}, {0.0F + m_xpos, 4.33F, 0.0F}, {0.0F, 1.0F, 0.0F});
          m_light.position = {-2.7f + m_xpos,
                              0.4f,
                              0.0F};
        }
        if (ImGui::Button("Save Image"))
        {
          m_shouldSaveImage = true;
        }
        changed = PropertyEditor::entry("speed", [&]
                                        { return ImGui::SliderFloat("#1", &m_speed, 0.0F, 10.0F); });

        PropertyEditor::end();
      }

      bool changed{false};
      if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
      {
        PropertyEditor::begin();
        if (PropertyEditor::treeNode("Light"))
        {
          changed |= PropertyEditor::entry("Position", [&]
                                           { return ImGui::DragFloat3("#1", &m_light.position.x); });

          changed |= PropertyEditor::entry("Intensity", [&]
                                           { return ImGui::SliderFloat("#1", &m_light.intensity, 0.0F, 1000.0F, "%.3f", ImGuiSliderFlags_Logarithmic); });
          changed |=
              PropertyEditor::entry("Radius", [&]
                                    { return ImGui::SliderFloat("#1", &m_light.radius, 0.0F, 1.0F); });
          PropertyEditor::treePop();
        }
        if (PropertyEditor::treeNode("Ray Tracer"))
        {
          changed |= PropertyEditor::entry("Depth", [&]
                                           { return ImGui::SliderInt("#1", &m_pushConst.maxDepth, 0, 20); });
          changed |=
              PropertyEditor::entry("Samples", [&]
                                    { return ImGui::SliderInt("#1", &m_pushConst.maxSamples, 1, 100); });
          PropertyEditor::treePop();
        }
        PropertyEditor::end();
      }

      if (ImGui::CollapsingHeader("Tonemapper"))
      {
        changed |= m_tonemapper->onUI();
      }

      ImGui::End();
      if (changed)
        ;
      // resetFrame();
    }

    { // Rendering Viewport

      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image(m_gBuffers->getDescriptorSet(eImgTonemapped),
                   m_auto_render ? ImVec2{1920, 1080} : ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    auto sdbg = m_dutil->DBG_SCOPE(cmd);

    // anim
    {
      const int idx = 1;
      int currentFrame = m_frame / SUB_FRAMES;
      m_nodes[idx].translation = vec3f(1.0f, 1.0f + sin(m_frame * (1.0f / 100.0f)), 1.0f);

      VkAccelerationStructureInstanceKHR &tinst = m_tlas[idx];

      { // Door
        m_nodes[0].translation = {-3.0f + m_xpos,
                                  0.7f,
                                  0.0F};
        VkAccelerationStructureInstanceKHR &tinst = m_tlas[0];
        tinst.transform = nvvk::toTransformMatrixKHR(m_nodes[0].localMatrix());

        m_nodes[1].translation = {-2.0f + m_xpos,
                                  1.55f,
                                  0.0F};
        VkAccelerationStructureInstanceKHR &tinstSp = m_tlas[1];
        tinstSp.transform = nvvk::toTransformMatrixKHR(m_nodes[1].localMatrix());
      }

      // tinst.transform = nvvk::toTransformMatrixKHR(m_nodes[idx].localMatrix());
      float speed = 10.0f;
      { // Gate 1
        const int i1 = 6;
        const int i2 = 7;
        const int i3 = 8;

        m_nodes[i1].rotation = nvmath::slerp_quats(
            currentFrame * (m_speed / 100.0f),
            nvmath::quatf(0.0f, 0.0f, 0.0f, 1.0f),
            nvmath::quatf(0.259f, 0.0f, 0.0f, 0.966f));
        VkAccelerationStructureInstanceKHR &tinst1 = m_tlas[i1];
        tinst1.transform = nvvk::toTransformMatrixKHR(m_nodes[i1].localMatrix());

        m_nodes[i2].rotation = nvmath::slerp_quats(
            currentFrame * (m_speed / 100.0f),
            nvmath::quatf(0.500f, 0.0f, 0.0f, 0.866f),
            nvmath::quatf(0.707f, 0.0f, 0.0f, 0.707f));
        VkAccelerationStructureInstanceKHR &tinst2 = m_tlas[i2];
        tinst2.transform = nvvk::toTransformMatrixKHR(m_nodes[i2].localMatrix());

        m_nodes[i3].rotation = nvmath::slerp_quats(
            currentFrame * (m_speed / 100.0f),
            nvmath::quatf(0.500f, 0.0f, 0.0f, -0.866f),
            nvmath::quatf(0.259f, 0.0f, 0.0f, -0.966f));
        VkAccelerationStructureInstanceKHR &tinst3 = m_tlas[i3];
        tinst3.transform = nvvk::toTransformMatrixKHR(m_nodes[i3].localMatrix());
      }

      { // Gate 2
        const int i1 = 12;
        const int i2 = 13;
        const int i3 = 14;
        const float offset = -1.0f;
        m_nodes[i1].rotation = nvmath::slerp_quats(
            currentFrame * (m_speed / 100.0f) + offset,
            nvmath::quatf(0.0f, 0.0f, 0.0f, 1.0f),
            nvmath::quatf(0.259f, 0.0f, 0.0f, 0.966f));
        VkAccelerationStructureInstanceKHR &tinst1 = m_tlas[i1];
        tinst1.transform = nvvk::toTransformMatrixKHR(m_nodes[i1].localMatrix());

        m_nodes[i2].rotation = nvmath::slerp_quats(
            currentFrame * (m_speed / 100.0f) + offset,
            nvmath::quatf(0.500f, 0.0f, 0.0f, 0.866f),
            nvmath::quatf(0.707f, 0.0f, 0.0f, 0.707f));
        VkAccelerationStructureInstanceKHR &tinst2 = m_tlas[i2];
        tinst2.transform = nvvk::toTransformMatrixKHR(m_nodes[i2].localMatrix());

        m_nodes[i3].rotation = nvmath::slerp_quats(
            currentFrame * (m_speed / 100.0f) + offset,
            nvmath::quatf(0.500f, 0.0f, 0.0f, -0.866f),
            nvmath::quatf(0.259f, 0.0f, 0.0f, -0.966f));
        VkAccelerationStructureInstanceKHR &tinst3 = m_tlas[i3];
        tinst3.transform = nvvk::toTransformMatrixKHR(m_nodes[i3].localMatrix());
      }
      { // Mirror
        const int m1 = 15;
        const float offset = -2.0f;
        m_nodes[m1].rotation = nvmath::slerp_quats(
            currentFrame * (m_speed / 100.0f) + offset,
            nvmath::quatf(0.0f, 0.0f, 0.0f, 1.0f),
            nvmath::quatf(0.707f, 0.0f, 0.0f, -0.707f));
        VkAccelerationStructureInstanceKHR &tinst1 = m_tlas[m1];
        tinst1.transform = nvvk::toTransformMatrixKHR(m_nodes[m1].localMatrix());
      }
    }

    m_rtBuilder.buildTlas(m_tlas, m_rtFrags, true);
    if (!updateFrame())
    {
      return;
    }
    if (m_shaderCompile)
    {
      m_shaderCompile = false;

      reloadPipeline();
      m_gbufferPass->createComputePIpeline();
      m_temporalPathReusePass->createComputePipeline();
      m_spatialPathReusePass->createComputePipeline();
      m_resVisualizer->createComputePipeline();
    }

    // current screen size
    const auto &size = m_app->getViewportSize();

    float view_aspect_ratio = m_viewSize.x / m_viewSize.y;
    nvmath::vec3f eye;
    nvmath::vec3f center;
    nvmath::vec3f up;
    CameraManip.getLookat(eye, center, up);

    // Update Frame buffer uniform buffer
    const auto &clip = CameraManip.getClipPlanes();
    {
      m_frameinfo.prevProj = m_frameinfo.proj;
      m_frameinfo.prevView = m_frameinfo.view;
      m_frameinfo.prevProjInv = m_frameinfo.projInv;
      m_frameinfo.prevViewInv = m_frameinfo.viewInv;
      m_frameinfo.proj = nvmath::perspectiveVK(CameraManip.getFov(), view_aspect_ratio, clip.x, clip.y);
      m_frameinfo.view = CameraManip.getMatrix();
      m_frameinfo.projInv = nvmath::inverse(m_frameinfo.proj);
      m_frameinfo.viewInv = nvmath::inverse(m_frameinfo.view);
      m_frameinfo.camPos = eye;
      m_frameinfo.width = size.width;
      m_frameinfo.height = size.height;
      m_frameinfo.frame = m_frame;
    }
    // m_frameinfo{
    //     .proj =
    //         .view = CameraManip.getMatrix(),
    //     .projInv = nvmath::inverse(finfo.proj),
    //     .viewInv = nvmath::inverse(finfo.view),
    //     .camPos = eye,
    // };
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(FrameInfo), &m_frameinfo);
    m_pushConst.subFrame = m_subframe;
    m_pushConst.frame = m_frame;
    m_pushConst.light = m_light;

    VkMemoryBarrier memBarrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier,
                         0, nullptr, 0, nullptr);

    updatePassDescriptors();
    m_resVisualizer->updatePushConstant(m_pushConst);

    // Raytraced GBuffer
    m_gbufferPass->runCompute(cmd, size);
    // GBuffer barrier
    {
      VkBufferMemoryBarrier2KHR buffer_barrier{};
      buffer_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
      buffer_barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
      buffer_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT_KHR;
      buffer_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
      buffer_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
      buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      buffer_barrier.buffer = m_gbufferContainer->getGBuffer().buffer;
      buffer_barrier.size = m_gbufferContainer->getBufferSize();

      VkDependencyInfoKHR dep_info{};
      dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
      dep_info.bufferMemoryBarrierCount = 1;
      dep_info.pBufferMemoryBarriers = &buffer_barrier;
      vkCmdPipelineBarrier2KHR(cmd, &dep_info);
    }

    // Initial Sampling
    std::vector<VkDescriptorSet> descSets{m_rtSet->getSet()};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipe.plines[0]);
    pushDescriptorSet(cmd);
    vkCmdPushConstants(cmd, m_rtPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant), &m_pushConst);

    vkCmdDispatch(cmd, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);

    // Initial Sampling Reservoir Barrier
    {
      std::vector<VkBufferMemoryBarrier2KHR> barriers;

      VkBufferMemoryBarrier2KHR buffer_barrier{};
      buffer_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
      buffer_barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
      buffer_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT_KHR;
      buffer_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
      buffer_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
      buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      buffer_barrier.buffer = m_diResContainer->getReservoir().buffer;
      buffer_barrier.size = m_diResContainer->getBufferSize();

      VkBufferMemoryBarrier2KHR reservoir_barrier{};
      reservoir_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
      reservoir_barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
      reservoir_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT_KHR;
      reservoir_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
      reservoir_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
      reservoir_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      reservoir_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      reservoir_barrier.buffer = m_initReservoirContainer->getReservoir().buffer;
      reservoir_barrier.size = m_initReservoirContainer->getBufferSize();

      barriers.emplace_back(buffer_barrier);
      barriers.emplace_back(reservoir_barrier);

      VkDependencyInfoKHR dep_info{};
      dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
      dep_info.bufferMemoryBarrierCount = barriers.size();
      dep_info.pBufferMemoryBarriers = barriers.data();
      vkCmdPipelineBarrier2KHR(cmd, &dep_info);
    }

    m_temporalPathReusePass->runCompute(cmd, size);
    {
      std::vector<VkBufferMemoryBarrier2KHR> barriers;

      VkBufferMemoryBarrier2KHR buffer_barrier{};
      buffer_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
      buffer_barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
      buffer_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT_KHR;
      buffer_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
      buffer_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
      buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      buffer_barrier.buffer = m_spatialReservoirContainer->getReservoir().buffer;
      buffer_barrier.size = m_spatialReservoirContainer->getBufferSize();

      // VkBufferMemoryBarrier2KHR reservoir_barrier{};
      // reservoir_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
      // reservoir_barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
      // reservoir_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT_KHR;
      // reservoir_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
      // reservoir_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
      // reservoir_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      // reservoir_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      // reservoir_barrier.buffer = m_initReservoirContainer->getReservoir().buffer;
      // reservoir_barrier.size = m_initReservoirContainer->getBufferSize();

      barriers.emplace_back(buffer_barrier);
      // barriers.emplace_back(reservoir_barrier);

      VkDependencyInfoKHR dep_info{};
      dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
      dep_info.bufferMemoryBarrierCount = 1;
      dep_info.pBufferMemoryBarriers = &buffer_barrier;
      vkCmdPipelineBarrier2KHR(cmd, &dep_info);

      auto image_memory_barrier =
          nvvk::makeImageMemoryBarrier(m_gBuffers->getColorImage(eImgIntermediate1), VK_ACCESS_SHADER_READ_BIT,
                                       VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                           nullptr, 0, nullptr, 1, &image_memory_barrier);
    }

    m_spatialPathReusePass->runCompute(cmd, size);

    {
      VkBufferMemoryBarrier2KHR reservoir_barrier{};
      reservoir_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
      reservoir_barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
      reservoir_barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT_KHR;
      reservoir_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
      reservoir_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
      reservoir_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      reservoir_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      reservoir_barrier.buffer = m_initReservoirContainer->getReservoir().buffer;
      reservoir_barrier.size = m_initReservoirContainer->getBufferSize();

      VkDependencyInfoKHR dep_info{};
      dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
      dep_info.bufferMemoryBarrierCount = 1;
      dep_info.pBufferMemoryBarriers = &reservoir_barrier;
      vkCmdPipelineBarrier2KHR(cmd, &dep_info);

      auto image_memory_barrier =
          nvvk::makeImageMemoryBarrier(m_gBuffers->getColorImage(eImgIntermediate1), VK_ACCESS_SHADER_READ_BIT,
                                       VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                           nullptr, 0, nullptr, 1, &image_memory_barrier);
    }

    m_resVisualizer->runCompute(cmd, size);

    // Making sure the rendered image is ready to be used
    auto image_memory_barrier =
        nvvk::makeImageMemoryBarrier(m_gBuffers->getColorImage(eImgRendered), VK_ACCESS_SHADER_READ_BIT,
                                     VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &image_memory_barrier);

    m_tonemapper->runCompute(cmd, size);

    // int imageNum = ;
    if (m_shouldSaveImage || (m_auto_render && (m_frame % SUB_FRAMES == SUB_FRAMES - 1)))
    {
      auto image_memory_barrier =
          nvvk::makeImageMemoryBarrier(m_gBuffers->getColorImage(eImgTonemapped), VK_ACCESS_SHADER_READ_BIT,
                                       VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                           nullptr, 0, nullptr, 1, &image_memory_barrier);
      if (m_saveImageJobs.size() > 300 && m_saveImageJobs.front().joinable())
      {
        m_saveImageJobs.front().join();
        m_saveImageJobs.pop();
      }
      std::string formattedNum = std::to_string(m_anim_count);
      while (formattedNum.length() < 3)
      {
        formattedNum = "0" + formattedNum;
      }
      callSaveImageJob(formattedNum + ".jpg");
      m_anim_count++;
      if (m_auto_render && m_anim_count > 150)
      {
        while (!m_saveImageJobs.empty())
        {
          m_saveImageJobs.front().join();
          m_saveImageJobs.pop();
        }
        m_app->close();
      }
    }
  }

  void callSaveImageJob(const std::string &outFilename)
  {
    const nvh::ScopedTimer s_timer("Save Image\n");
    // Create a temporary buffer to hold the pixels of the image
    const VkBufferUsageFlags usage{VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    const VkDeviceSize buffer_size = 4 * sizeof(uint8_t) * m_gBuffers->getSize().width * m_gBuffers->getSize().height;
    nvvk::Buffer pixel_buffer = m_alloc->createBuffer(buffer_size, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    imageToBuffer(m_gBuffers->getColorImage(eImgTonemapped), pixel_buffer.buffer);
    std::thread th(&RayQuery::saveImage, this, pixel_buffer, outFilename);

    m_saveImageJobs.push(std::move(th));
  }

private:
  void saveImage(nvvk::Buffer pixel_buffer, const std::string &outFilename)
  {
    std::filesystem::path path = std::filesystem::current_path();
    auto filePath = path.parent_path().string() + "/" + outFilename;
    // Write the buffer to disk
    LOGI(" - Size: %d, %d\n", m_gBuffers->getSize().width, m_gBuffers->getSize().height);
    LOGI(" - Bytes: %d\n", m_gBuffers->getSize().width * m_gBuffers->getSize().height * 4);
    LOGI(" - Out name: %s\n", filePath.c_str());
    const void *data = m_alloc->map(pixel_buffer);

    stbi_write_jpg(filePath.c_str(), m_gBuffers->getSize().width, m_gBuffers->getSize().height, 4, data, 0);
    m_alloc->unmap(pixel_buffer);

    // Destroy temporary buffer
    m_alloc->destroy(pixel_buffer);
    m_shouldSaveImage = false;
  }
  //--------------------------------------------------------------------------------------------------
  // Copy the image to a buffer - this linearize the image memory
  //
  void imageToBuffer(const VkImage &imgIn, const VkBuffer &pixelBufferOut)
  {
    const nvh::ScopedTimer s_timer(" - Image To Buffer");

    auto *cmd = m_app->createTempCmdBuffer();

    // Make the image layout eTransferSrcOptimal to copy to buffer
    const VkImageSubresourceRange subresource_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    nvvk::cmdBarrierImageLayout(cmd, imgIn, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresource_range);

    // Copy the image to the buffer
    VkBufferImageCopy copy_region{};
    copy_region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy_region.imageExtent = VkExtent3D{m_gBuffers->getSize().width, m_gBuffers->getSize().height, 1};
    vkCmdCopyImageToBuffer(cmd, imgIn, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pixelBufferOut, 1, &copy_region);

    // Put back the image as it was
    nvvk::cmdBarrierImageLayout(cmd, imgIn, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, subresource_range);
    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  void createScene()
  {

    m_materials.push_back({{0.985f, 0.862f, 0.405f}, 0.5f, 0.0f});
    m_materials.push_back({{0.622f, 0.928f, 0.728f}, 0.05f, 1.0f});
    m_materials.push_back({{.7F, .7F, .7F}, 0.3f, 0.0f});
    m_materials.push_back({{0.125, 0.0, 0.301}, 0.4f, 1.0f}); // Black violet
    m_materials.push_back({{1.0, 1.0, 1.0}, 1.0f, 0.0f});     // White
    m_materials.push_back({{0.0, 0.0, 0.1}, 0.4f, 0.0f});     // mat black
    m_materials.push_back({{1.0, 1.0, 1.0}, 0.0f, 0.0f});     // White with reflection 6
    m_materials.push_back({{1.0, 1.0, 1.0}, 0.0f, 1.0f});     // Mirror

    m_meshes.emplace_back(nvh::createCube(2.8, 0.2, 6));
    m_meshes.emplace_back(nvh::createSphereUv(0.18f));
    m_meshes.emplace_back(nvh::createPlane(10, 100, 100));
    m_meshes.emplace_back(nvh::createCube(20, 0.5f, 20));

    m_meshes.emplace_back(nvh::createCube(100, 20, 1)); // Right Wall
    m_meshes.emplace_back(nvh::createCube(100, 20, 1)); // Left Wall

    m_meshes.emplace_back(nvh::createCube(0.1, 60, 12)); // Panel 1
    m_meshes.emplace_back(nvh::createCube(0.1, 60, 12)); // Panel 2
    m_meshes.emplace_back(nvh::createCube(0.1, 60, 12)); // Panel 3

    m_meshes.emplace_back(nvh::createCube(15, 20, 1.1));  // Room1 R
    m_meshes.emplace_back(nvh::createCube(15, 20, 1.1));  // Room1 L
    m_meshes.emplace_back(nvh::createCube(15, 1.05, 20)); // Room1 B 11

    m_meshes.emplace_back(nvh::createCube(0.1, 60, 12)); // Panel 1
    m_meshes.emplace_back(nvh::createCube(0.1, 60, 12)); // Panel 2
    m_meshes.emplace_back(nvh::createCube(0.1, 60, 12)); // Panel 3 14

    m_meshes.emplace_back(nvh::createCube(0.1, 40, 30)); // Mirror1
    m_meshes.emplace_back(nvh::createCube(0.1, 20, 15)); // Mirror 2
    m_meshes.emplace_back(nvh::createCube(0.1, 40, 30)); // Mirror 3 17

    // Instance Cube
    {
      auto &n = m_nodes.emplace_back();
      n.mesh = 0;
      n.material = 0;
      n.translation = {-3.0f, 0.7f, 0.0F};
      n.rotation = nvmath::quatf(0.0f, 0.0f, 0.259f, 0.966f);
    }

    // Instance Sphere
    {
      auto &n = m_nodes.emplace_back();
      n.mesh = 1;
      n.material = 1;
      n.translation = {-2.0f, 1.6f, 0.0F};
    }

    // Adding a plane & material
    {
      auto &n = m_nodes.emplace_back();
      n.mesh = 2;
      n.material = 3;
      n.translation = {0.0f, 0.0f, 0.0f};
    }

    {
      auto &n = m_nodes.emplace_back();
      n.mesh = 3;
      n.material = 2;
      n.translation = {0.0f, 30.0f, 0.0f};
    }

    { // Right Wall
      auto &n = m_nodes.emplace_back();
      n.mesh = 4;
      n.material = 3;
      // nvmath::vec3f()
      n.translation = {0.0f, 8.66f, 5.0f};
      n.rotation = nvmath::quatf(-0.259f, 0.0f, 0.0f, 0.966f);
    }
    { // Left Wall
      auto &n = m_nodes.emplace_back();
      n.mesh = 5;
      n.material = 3;
      n.translation = {0.0f, 8.66f, -5.0f};
      n.rotation = nvmath::quatf(0.259f, 0.0f, 0.0f, 0.966f);
    }

    { // Gate Panel 1
      auto &n = m_nodes.emplace_back();
      n.mesh = 6;
      n.material = 5;
      n.translation = {0.0f, 17.7f, -6.0f};
      // n.rotation = nvmath::quatf(0.0f, 0.0f, 0.0f, 1.0f);  // from
      n.rotation = nvmath::quatf(0.259f, 0.0f, 0.0f, 0.966f); // to
    }

    { // Gate Panel 2
      auto &n = m_nodes.emplace_back();
      n.mesh = 7;
      n.material = 5;
      n.translation = {0.0f, -6.7f, -8.66f};
      n.rotation = nvmath::quatf(0.500f, 0.0f, 0.0f, 0.866f); // from
      n.rotation = nvmath::quatf(0.707f, 0.0f, 0.0f, 0.707f); // to
    }

    { // Gate Panel 3
      auto &n = m_nodes.emplace_back();
      n.mesh = 8;
      n.material = 5;
      n.translation = {0.0f, 5.3f, 13.215f};
      n.rotation = nvmath::quatf(0.500f, 0.0f, 0.0f, -0.866f); // from
      n.rotation = nvmath::quatf(0.259f, 0.0f, 0.0f, -0.966f); // to
    }

    { // Room1 R
      auto &n = m_nodes.emplace_back();
      n.mesh = 9;
      n.material = 4;
      n.translation = {10.0f, 8.66f, 4.95f};
      n.rotation = nvmath::quatf(-0.259f, 0.0f, 0.0f, 0.966f);
    }

    { // Room1 L
      auto &n = m_nodes.emplace_back();
      n.mesh = 10;
      n.material = 4;
      n.translation = {10.0f, 8.66f, -4.95f};
      n.rotation = nvmath::quatf(0.259f, 0.0f, 0.0f, 0.966f);
    }

    { // Room1 B
      auto &n = m_nodes.emplace_back();
      n.mesh = 11;
      n.material = 4;
      n.translation = {10.0f, -0.5f, 0.0f};
      // n.rotation = nvmath::quatf(-0.259f, 0.0f, 0.0f, 0.966f);
    }

    { // Gate Panel 1
      auto &n = m_nodes.emplace_back();
      n.mesh = 12;
      n.material = 6;
      n.translation = {15.0f, 17.7f, -6.0f};
      n.rotation = nvmath::quatf(0.0f, 0.0f, 0.0f, 1.0f); // from
      // n.rotation = nvmath::quatf(0.259f, 0.0f, 0.0f, 0.966f); // to
    }

    { // Gate Panel 2
      auto &n = m_nodes.emplace_back();
      n.mesh = 13;
      n.material = 6;
      n.translation = {15.0f, -6.7f, -8.66f};
      n.rotation = nvmath::quatf(0.500f, 0.0f, 0.0f, 0.866f); // from
      // n.rotation = nvmath::quatf(0.707f, 0.0f, 0.0f, 0.707f); // to
    }

    { // Gate Panel 3
      auto &n = m_nodes.emplace_back();
      n.mesh = 14;
      n.material = 6;
      n.translation = {15.0f, 5.3f, 13.215f};
      n.rotation = nvmath::quatf(0.500f, 0.0f, 0.0f, -0.866f); // from
      // n.rotation = nvmath::quatf(0.259f, 0.0f, 0.0f, -0.966f); // to
    }

    { // Mirror 1
      auto &n = m_nodes.emplace_back();
      n.mesh = 15;
      n.material = 7;
      n.translation = {25.0f, 0.01f, 0.0f};
      n.rotation = nvmath::quatf(0.0f, 0.0f, 0.0f, 1.0f); // from
      // n.rotation = nvmath::quatf(0.707f, 0.0f, 0.0f, -0.707f); // to
    }
    // { // MIrror 2
    //   auto &n = m_nodes.emplace_back();
    //   n.mesh = 16;
    //   n.material = 7;
    //   n.translation = {24.99f, 8.66f, -4.99f};
    //   n.rotation = nvmath::quatf(0.500f, 0.0f, 0.0f, -0.866f);      // from
    //   n.rotation = nvmath::quatf(0.612f, -0.354f, 0.612f, -0.354f); // to
    // }
    // { // Mirror 3
    //   auto &n = m_nodes.emplace_back();
    //   n.mesh = 17;
    //   n.material = 7;
    //   n.translation = {24.98f, 8.66f, 5.0f};
    //   n.rotation = nvmath::quatf(0.0f, 0.0f, 0.0f, 1.0f); // from
    //   // n.rotation = nvmath::quatf(0.259f, 0.0f, 0.0f, -0.966f); // to
    // }

    m_light.intensity = 500.0f;
    m_light.position = {-2.7f, 0.4f, 0.0f};
    m_light.radius = 0.1f;

    // Setting camera to see the scene
    CameraManip.setClipPlanes({0.1F, 100.0F});
    CameraManip.setLookat({-15.0F, 4.33F, 0.0f}, {0.0F, 4.33F, 0.0F}, {0.0F, 1.0F, 0.0F});
    // Default parameters for overall material
    m_pushConst.maxDepth = 5;
    m_pushConst.frame = 0;
    m_pushConst.fireflyClampThreshold = 10;
    // m_pushConst.maxSamples = m_auto_render ? 30 : 30;
    m_pushConst.light = m_light;
    m_pushConst.maxSubframes = 100;
    m_pushConst.subFrame = 0;
  }

  void createGbuffers(const nvmath::vec2f &size)
  {
    // Rendering image targets
    m_viewSize = size;
    std::vector<VkFormat> color_buffers = {m_outColorFormat, m_colorFormat, m_colorFormat}; // tonemapped, original
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(),
                                                   VkExtent2D{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)},
                                                   color_buffers, m_depthFormat);
  }

  // Create all Vulkan buffer data
  void createVkBuffers()
  {
    auto *cmd = m_app->createTempCmdBuffer();
    m_bMeshes.resize(m_meshes.size());

    auto rtUsageFlag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    // Create a buffer of Vertex and Index per mesh
    std::vector<PrimMeshInfo> primInfo;
    for (size_t i = 0; i < m_meshes.size(); i++)
    {
      auto &m = m_bMeshes[i];
      m.vertices = m_alloc->createBuffer(cmd, m_meshes[i].vertices, rtUsageFlag);
      m.indices = m_alloc->createBuffer(cmd, m_meshes[i].triangles, rtUsageFlag);
      m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
      m_dutil->DBG_NAME_IDX(m.indices.buffer, i);

      // To find the buffers of the mesh (buffer reference)
      PrimMeshInfo info{
          .vertexAddress = nvvk::getBufferDeviceAddress(m_device, m.vertices.buffer),
          .indexAddress = nvvk::getBufferDeviceAddress(m_device, m.indices.buffer),
      };
      primInfo.emplace_back(info);
    }

    // Creating the buffer of all primitive information
    m_bPrimInfo = m_alloc->createBuffer(cmd, primInfo, rtUsageFlag);
    m_dutil->DBG_NAME(m_bPrimInfo.buffer);

    // Create the buffer of the current frame, changing at each frame
    m_bFrameInfo = m_alloc->createBuffer(sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bFrameInfo.buffer);

    // Primitive instance information
    std::vector<InstanceInfo> instInfo;
    for (auto &node : m_nodes)
    {
      InstanceInfo info{
          info.transform = node.localMatrix(),
          info.materialID = node.material,
      };
      instInfo.emplace_back(info);
    }
    m_bInstInfoBuffer =
        m_alloc->createBuffer(cmd, instInfo, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bInstInfoBuffer.buffer);

    m_bMaterials = m_alloc->createBuffer(cmd, m_materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bMaterials.buffer);

    // Buffer references of all scene elements
    SceneDescription sceneDesc{
        .materialAddress = nvvk::getBufferDeviceAddress(m_device, m_bMaterials.buffer),
        .instInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_bInstInfoBuffer.buffer),
        .primInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_bPrimInfo.buffer),
        .light = m_light,
    };

    m_bSceneDesc = m_alloc->createBuffer(cmd, sizeof(SceneDescription), &sceneDesc,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bSceneDesc.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  //--------------------------------------------------------------------------------------------------
  // Converting a PrimitiveMesh as input for BLAS
  //
  nvvk::RaytracingBuilderKHR::BlasInput primitiveToGeometry(const nvh::PrimitiveMesh &prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress)
  {
    uint32_t maxPrimitiveCount = static_cast<uint32_t>(prim.triangles.size());

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangles.vertexFormat = VK_FORMAT_R32G32B32A32_SFLOAT; // vec3 vertex position data.
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride = sizeof(nvh::PrimitiveVertex);
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexAddress;
    triangles.maxVertex = static_cast<uint32_t>(prim.vertices.size());
    // triangles.transformData; // Identity

    // Identify the above data as containing opaque triangles.
    VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    asGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    asGeom.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    asGeom.geometry.triangles = triangles;

    VkAccelerationStructureBuildRangeInfoKHR offset{};
    offset.firstVertex = 0;
    offset.primitiveCount = maxPrimitiveCount;
    offset.primitiveOffset = 0;
    offset.transformOffset = 0;

    // Our BLAS is made from only one geometry, but could be made of many geometries
    nvvk::RaytracingBuilderKHR::BlasInput input;
    input.asGeometry.emplace_back(asGeom);
    input.asBuildOffsetInfo.emplace_back(offset);

    return input;
  }

  //--------------------------------------------------------------------------------------------------
  // Create all bottom level acceleration structures (BLAS)
  //
  void createBottomLevelAS()
  {
    // BLAS - Storing each primitive in a geometry
    std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
    allBlas.reserve(m_meshes.size());

    for (uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      auto vertexAddress = nvvk::getBufferDeviceAddress(m_device, m_bMeshes[p_idx].vertices.buffer);
      auto indexAddress = nvvk::getBufferDeviceAddress(m_device, m_bMeshes[p_idx].indices.buffer);

      auto geo = primitiveToGeometry(m_meshes[p_idx], vertexAddress, indexAddress);
      allBlas.push_back({geo});
    }
    m_rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    m_tlas.reserve(m_nodes.size());
    for (auto &node : m_nodes)
    {
      VkGeometryInstanceFlagsKHR flags{VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV};

      VkAccelerationStructureInstanceKHR rayInst{};
      rayInst.transform = nvvk::toTransformMatrixKHR(node.localMatrix()); // Position of the instance
      rayInst.instanceCustomIndex = node.mesh;                            // gl_InstanceCustomIndexEXT
      rayInst.accelerationStructureReference = m_rtBuilder.getBlasDeviceAddress(node.mesh);
      rayInst.instanceShaderBindingTableRecordOffset = 0; // We will use the same hit group for all objects
      rayInst.flags = flags;
      rayInst.mask = 0xFF;
      m_tlas.emplace_back(rayInst);
    }
    m_rtFrags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    m_rtBuilder.buildTlas(m_tlas, m_rtFrags);
  }

  void reloadPipeline()
  {
    auto spirvCode = m_compiler->compile(L"ray_query.hlsl", L"computeMain");
    VkPipelineShaderStageCreateInfo stageInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = nvvk::createShaderModule(m_device,
                                           static_cast<const uint32_t *>(spirvCode->GetBufferPointer()), spirvCode->GetBufferSize()),
        .pName = "computeMain",
    };

    VkComputePipelineCreateInfo cpCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stageInfo,
        .layout = m_rtPipe.layout,
    };

    vkCreateComputePipelines(m_device, {}, 1, &cpCreateInfo, nullptr, &m_rtPipe.plines[0]);

    vkDestroyShaderModule(m_device, cpCreateInfo.stage.module, nullptr);
  }
  //--------------------------------------------------------------------------------------------------
  // Creating the pipeline: shader ...
  //
  void createCompPipelines()
  {
    m_rtPipe.destroy(m_device);
    m_rtSet->deinit();
    m_rtSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_rtPipe.plines.resize(1);

    // This descriptor set, holds the top level acceleration structure and the output image
    m_rtSet->addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_outBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_rtSet->addBinding(B_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_sceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_gbuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_rtSet->addBinding(B_outReservoir, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_rtSet->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

    // pushing time
    VkPushConstantRange pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant)};
    VkPipelineLayoutCreateInfo plCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1U,
        .pSetLayouts = &m_rtSet->getLayout(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstant,
    };
    vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_rtPipe.layout);

    reloadPipeline();
  }

  void pushDescriptorSet(VkCommandBuffer cmd)
  {
    // Write to descriptors
    VkAccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();
    VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    descASInfo.accelerationStructureCount = 1;
    descASInfo.pAccelerationStructures = &tlas;
    VkDescriptorImageInfo imageInfo{{}, m_gBuffers->getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo sceneDesc{m_bSceneDesc.buffer, 0, VK_WHOLE_SIZE};
    auto bufInfo = m_diResContainer->getReservoir();
    auto resInfo = m_initReservoirContainer->getReservoir();
    auto gbufInfo = m_gbufferContainer->getGBuffer();
    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_rtSet->makeWrite(0, B_tlas, &descASInfo));
    writes.emplace_back(m_rtSet->makeWrite(0, B_outImage, &imageInfo));
    writes.emplace_back(m_rtSet->makeWrite(0, B_outBuffer, &bufInfo));
    writes.emplace_back(m_rtSet->makeWrite(0, B_frameInfo, &dbi_unif));
    writes.emplace_back(m_rtSet->makeWrite(0, B_sceneDesc, &sceneDesc));
    writes.emplace_back(m_rtSet->makeWrite(0, B_gbuffer, &gbufInfo));
    writes.emplace_back(m_rtSet->makeWrite(0, B_outReservoir, &resInfo));

    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipe.layout, 0,
                              static_cast<uint32_t>(writes.size()), writes.data());
  }

  //--------------------------------------------------------------------------------------------------
  // To be call when renderer need to re-start
  //
  void resetFrame()
  {
    // m_frame = -1;
  }

  //--------------------------------------------------------------------------------------------------
  // If the camera matrix has changed, resets the frame.
  // otherwise, increments frame.
  //
  bool updateFrame()
  {
    static float ref_fov{0};
    static float ref_cam_matrix[16];

    const auto &m = CameraManip.getMatrix();
    const auto fov = CameraManip.getFov();

    if (memcmp(&ref_cam_matrix[0], &m.a00, sizeof(nvmath::mat4f)) != 0 || ref_fov != fov)
    {
      resetFrame();
      memcpy(&ref_cam_matrix[0], &m.a00, sizeof(nvmath::mat4f));
      ref_fov = fov;
    }

    if (m_frame >= m_maxFrames)
    {
      return false;
    }
    m_frame++;
    m_subframe++;
    if (m_subframe > m_pushConst.maxSubframes)
    {
      m_subframe = 0;
    }
    m_temporalReservoirContainer.swap(m_spatialReservoirContainer);
    return true;
  }

  void destroyResources()
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    for (auto &m : m_bMeshes)
    {
      m_alloc->destroy(m.vertices);
      m_alloc->destroy(m.indices);
    }
    m_alloc->destroy(m_bFrameInfo);
    m_alloc->destroy(m_bPrimInfo);
    m_alloc->destroy(m_bSceneDesc);
    m_alloc->destroy(m_bInstInfoBuffer);
    m_alloc->destroy(m_bMaterials);

    m_rtSet->deinit();
    m_gBuffers.reset();

    m_rtPipe.destroy(m_device);

    m_sbt.destroy();
    m_rtBuilder.destroy();
    // m_tonemapper.reset();
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application *m_app{nullptr};
  std::unique_ptr<HLSLShaderCompiler> m_compiler;
  std::unique_ptr<nvvk::DebugUtil> m_dutil;
  std::unique_ptr<nvvkhl::AllocVma> m_alloc;
  std::unique_ptr<nvvk::DescriptorSetContainer> m_rtSet; // Descriptor set
  std::unique_ptr<VisualizeReservoir> m_resVisualizer;
  std::unique_ptr<nvvkhl::TonemapperPostProcess> m_tonemapper;
  std::unique_ptr<GBuffer> m_gbufferPass;
  std::unique_ptr<PathReuse> m_spatialPathReusePass;
  std::unique_ptr<PathReuse> m_temporalPathReusePass;

  nvmath::vec2f m_viewSize = {1920, 1080};
  VkFormat m_colorFormat = VK_FORMAT_R32G32B32A32_SFLOAT; // Color format of the image
  VkFormat m_outColorFormat = VK_FORMAT_R8G8B8A8_UNORM;   // Color format of the image
  VkFormat m_depthFormat = VK_FORMAT_D32_SFLOAT;          // Depth format of the depth buffer
  VkDevice m_device = VK_NULL_HANDLE;                     // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;            // G-Buffers: color + depth

  // Resources
  std::vector<VkAccelerationStructureInstanceKHR> m_tlas;
  VkBuildAccelerationStructureFlagsKHR m_rtFrags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices; // Buffer of the vertices
    nvvk::Buffer indices;  // Buffer of the indices
  };
  std::vector<PrimitiveMeshVk> m_bMeshes;
  nvvk::Buffer m_bFrameInfo;
  nvvk::Buffer m_bPrimInfo;
  nvvk::Buffer m_bSceneMatrixInfo;
  nvvk::Buffer m_bSceneDesc; // SceneDescription
  nvvk::Buffer m_bInstInfoBuffer;
  nvvk::Buffer m_bMaterials;

  // Data and setting
  std::vector<nvh::PrimitiveMesh> m_meshes;
  std::vector<nvh::Node> m_nodes;
  std::vector<Material> m_materials;
  Light m_light;
  FrameInfo m_frameinfo;

  VkShaderModule m_initTraceModule = VK_NULL_HANDLE;

  // structuredbuffers
  std::unique_ptr<GBufferContainer> m_gbufferContainer;
  std::unique_ptr<DIReservoirContainer> m_diResContainer;
  std::unique_ptr<ReservoirContainer> m_initReservoirContainer;
  std::unique_ptr<ReservoirContainer> m_temporalReservoirContainer;
  std::unique_ptr<ReservoirContainer> m_spatialReservoirContainer;

  // Pipeline
  PushConstant m_pushConst{};                         // Information sent to the shader
  VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE; // The description of the pipeline
  VkPipeline m_graphicsPipeline = VK_NULL_HANDLE;     // The graphic pipeline to render
  int m_frame{0};
  int m_subframe{0};
  int m_maxFrames{1000000};

  float m_xpos = 0.0f;

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::SBTWrapper m_sbt; // Shading binding table wrapper
  nvvk::RaytracingBuilderKHR m_rtBuilder;
  nvvkhl::PipelineContainer m_rtPipe;

  bool m_shaderCompile = false;
  bool m_shouldSaveImage = false;
  bool m_fullscreen = false;
  bool m_auto_render = false;
  uint m_anim_count = 0;
  float m_speed = 1.0f;
  std::queue<std::thread> m_saveImageJobs;
};

//////////////////////////////////////////////////////////////////////////
///
///
///
auto main(int argc, char **argv) -> int
{
  const std::string logfile = std::string("log_") + std::string(PROJECT_NAME) + std::string(".txt");
  nvprintSetLogFileName(logfile.c_str());
  nvh::CommandLineParser parser("RTCamp9 Render");
  bool auto_render = false;
  uint spp = 1;
  parser.addArgument({"-a", "--auto"}, &auto_render, "本番");
  parser.addArgument({"-s", "--spp"}, &spp, "サンプル数");
  if (!parser.parse(argc, argv))
  {
    parser.printHelp();
    return 1;
  }

  nvvkhl::ApplicationCreateInfo spec;
  spec.name = PROJECT_NAME;
  spec.vSync = false;
  spec.vkSetup.apiMajor = 1;
  spec.vkSetup.apiMinor = 3;

  spec.vkSetup.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelFeature); // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rtPipelineFeature); // To use vkCmdTraceRaysKHR
  spec.vkSetup.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);                        // Required by ray tracing pipeline
  VkPhysicalDeviceShaderClockFeaturesKHR clockFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, false, &clockFeature);
  spec.vkSetup.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayqueryFeature);
  VkPhysicalDeviceSynchronization2FeaturesKHR sync2Feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME, false, &sync2Feature);

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);
  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementTesting>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());        // Menu / Quit
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>()); // Window title info
  app->addElement(std::make_shared<RayQuery>(auto_render, spp));

  app->run();
  app.reset();

  return test->errorCode();
}
