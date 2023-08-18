#pragma once

#include "visualize_reservoir.hpp"

#if USE_HLSL
#include "_autogen/ray_query_computeMain.spirv.h"
const auto &comp = std::vector<char>{std::begin(ray_query_computeMain), std::end(ray_query_computeMain)};
#elif USE_SLANG
#include "_autogen/ray_query_computeMain.spirv.h"
const auto &comp = std::vector<uint32_t>{std::begin(ray_query_computeMain), std::end(ray_query_computeMain)};
#endif

VisualizeReservoir::VisualizeReservoir(nvvk::Context *ctx, nvvkhl::AllocVma *alloc)
    : m_ctx(ctx), m_dutil(std::make_unique<nvvk::DebugUtil>(ctx->m_device)), m_dset(std::make_unique<nvvk::DescriptorSetContainer>(ctx->m_device))
{
  m_pushConst.dummy = 0;
}

VisualizeReservoir::~VisualizeReservoir()
{
  vkDestroyPipeline(m_ctx->m_device, m_pipeline, nullptr);
  m_dset->deinit();
}

void VisualizeReservoir::createComputePipeline()
{
  nvvk::DebugUtil dbg(m_ctx->m_device);
  m_dset->addBinding(eDebugPassInput, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_dset->addBinding(eDebugPassOutput, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_dset->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);
  m_dutil->DBG_NAME(m_dset->getLayout());

  VkPushConstantRange push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DBGConstant)};
  m_dset->initPipeLayout(1, &push_constant);
  m_dutil->DBG_NAME(m_dset->getPipeLayout());

  VkPipelineShaderStageCreateInfo stage_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_info.module = nvvk::createShaderModule(m_ctx->m_device, comp.data(), sizeof(comp.data()));
  stage_info.pName = "main";

  VkComputePipelineCreateInfo comp_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  comp_info.layout = m_dset->getPipeLayout();
  comp_info.stage = stage_info;

  vkCreateComputePipelines(m_ctx->m_device, {}, 1, &comp_info, nullptr, &m_pipeline);
  m_dutil->DBG_NAME(m_pipeline);
  vkDestroyShaderModule(m_ctx->m_device, comp_info.stage.module, nullptr);
}

void VisualizeReservoir::updateComputeDescriptorSets(VkDescriptorBufferInfo inReservoir, VkDescriptorImageInfo outImage)
{
  m_ibuffer = std::make_unique<VkDescriptorBufferInfo>(inReservoir);
  m_oimage = std::make_unique<VkDescriptorImageInfo>(outImage);
  m_writes.clear();
  m_writes.emplace_back(m_dset->makeWrite(0, eDebugPassInput, m_ibuffer.get()));
  m_writes.emplace_back(m_dset->makeWrite(0, eDebugPassOutput, m_oimage.get()));
}

void VisualizeReservoir::runCompute(VkCommandBuffer cmd, const VkExtent2D &size)
{
  auto sdbg = m_dutil->DBG_SCOPE(cmd);
  vkCmdPushConstants(cmd, m_dset->getPipeLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DBGConstant), &m_pushConst);
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dset->getPipeLayout(), 0, static_cast<uint32_t>(m_writes.size()), m_writes.data());
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
  vkCmdDispatch(cmd, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
}

bool VisualizeReservoir::onUI()
{
  bool changed = false;
  using namespace ImGuiH;
  PropertyEditor::begin();
  PropertyEditor::end();
  return changed;
}