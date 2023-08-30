#pragma once

#include "visualize_reservoir.hpp"

VisualizeReservoir::VisualizeReservoir(nvvk::Context *ctx, nvvkhl::AllocVma *alloc, HLSLShaderCompiler *compiler)
    : m_ctx(ctx), m_dutil(std::make_unique<nvvk::DebugUtil>(ctx->m_device)), m_dset(std::make_unique<nvvk::DescriptorSetContainer>(ctx->m_device)), m_compiler(compiler)
{
}

VisualizeReservoir::~VisualizeReservoir()
{
  vkDestroyPipeline(m_ctx->m_device, m_pipeline, nullptr);
  m_dset->deinit();
}

void VisualizeReservoir::createPipelineLayout()
{
  nvvk::DebugUtil dbg(m_ctx->m_device);
  m_dset->addBinding(eDebugPassInput, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_dset->addBinding(B_compose_giInput, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_dset->addBinding(B_compose_thpInput, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_dset->addBinding(eDebugPassOutput, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_dset->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);
  m_dutil->DBG_NAME(m_dset->getLayout());

  VkPushConstantRange push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant)};
  m_dset->initPipeLayout(1, &push_constant);
  m_dutil->DBG_NAME(m_dset->getPipeLayout());
}

void VisualizeReservoir::createComputePipeline()
{
  auto spirvCode = m_compiler->compile(L"debug_reservoir.hlsl", L"main");

  VkPipelineShaderStageCreateInfo stage_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = nvvk::createShaderModule(m_ctx->m_device,
                                         static_cast<const uint32_t *>(spirvCode->GetBufferPointer()), spirvCode->GetBufferSize()),
      .pName = "main",
  };

  VkComputePipelineCreateInfo comp_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  comp_info.layout = m_dset->getPipeLayout();
  comp_info.stage = stage_info;

  vkCreateComputePipelines(m_ctx->m_device, {}, 1, &comp_info, nullptr, &m_pipeline);
  m_dutil->DBG_NAME(m_pipeline);
  vkDestroyShaderModule(m_ctx->m_device, comp_info.stage.module, nullptr);
}

void VisualizeReservoir::updateComputeDescriptorSets(VkDescriptorBufferInfo inReservoir, VkDescriptorBufferInfo inGiReservoir, VkDescriptorImageInfo inThpImage, VkDescriptorImageInfo outImage)
{
  m_ibuffer = std::make_unique<VkDescriptorBufferInfo>(inReservoir);
  m_giReservoir = std::make_unique<VkDescriptorBufferInfo>(inGiReservoir);
  m_thpimage = std::make_unique<VkDescriptorImageInfo>(inThpImage);
  m_oimage = std::make_unique<VkDescriptorImageInfo>(outImage);
  m_writes.clear();
  m_writes.emplace_back(m_dset->makeWrite(0, eDebugPassInput, m_ibuffer.get()));
  m_writes.emplace_back(m_dset->makeWrite(0, B_compose_giInput, m_giReservoir.get()));
  m_writes.emplace_back(m_dset->makeWrite(0, B_compose_thpInput, m_thpimage.get()));
  m_writes.emplace_back(m_dset->makeWrite(0, eDebugPassOutput, m_oimage.get()));
}

void VisualizeReservoir::updatePushConstant(PushConstant constant)
{
  m_pushConst = constant;
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