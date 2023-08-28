#pragma once
#include "gbuffer.hpp"

static const auto shaderfileName = L"ray_gbuffer.hlsl";

GBuffer::GBuffer(nvvk::Context *ctx, nvvkhl::AllocVma *alloc, HLSLShaderCompiler *compiler)
    : m_ctx(ctx), m_dutil(std::make_unique<nvvk::DebugUtil>(ctx->m_device)), m_dset(std::make_unique<nvvk::DescriptorSetContainer>(ctx->m_device)), m_compiler(compiler)
{
    m_pushConst = GbufConst{};
}

GBuffer::~GBuffer()
{
    // TODO:
}

void GBuffer::createPipelineLayout()
{
    nvvk::DebugUtil dbg(m_ctx->m_device);
    m_dset->addBinding(B_gbuf_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    m_dset->addBinding(B_gbuf_outBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_dset->addBinding(B_gbuf_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dset->addBinding(B_gbuf_sceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dset->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

    // // pushing time
    VkPushConstantRange pushConstant{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GbufConst)};
    m_dset->initPipeLayout(1, &pushConstant);
    m_dutil->DBG_NAME(m_dset->getPipeLayout());
}

void GBuffer::createComputePIpeline()
{
    auto spirvCode = m_compiler->compile(shaderfileName, L"main");
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

void GBuffer::updateComputeDescriptorSets(VkAccelerationStructureKHR tlas, VkDescriptorBufferInfo outBuffer, VkDescriptorBufferInfo frameInfo, VkDescriptorBufferInfo sceneInfo)
{
    m_tlas = std::make_unique<VkAccelerationStructureKHR>(tlas);
    VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    descASInfo.accelerationStructureCount = 1;
    descASInfo.pAccelerationStructures = m_tlas.get();
    m_asinfo = std::make_unique<VkWriteDescriptorSetAccelerationStructureKHR>(descASInfo);
    m_outbuffer = std::make_unique<VkDescriptorBufferInfo>(outBuffer);
    m_frameinfo = std::make_unique<VkDescriptorBufferInfo>(frameInfo);
    m_sceneinfo = std::make_unique<VkDescriptorBufferInfo>(sceneInfo);
    m_writes.clear();
    m_writes.emplace_back(m_dset->makeWrite(0, B_gbuf_tlas, m_asinfo.get()));
    m_writes.emplace_back(m_dset->makeWrite(0, B_gbuf_outBuffer, m_outbuffer.get()));
    m_writes.emplace_back(m_dset->makeWrite(0, B_gbuf_frameInfo, m_frameinfo.get()));
    m_writes.emplace_back(m_dset->makeWrite(0, B_gbuf_sceneDesc, m_sceneinfo.get()));
}

void GBuffer::updatePushConstants(const VkExtent2D &size)
{
    m_pushConst.height = size.height;
    m_pushConst.width = size.width;
}

void GBuffer::runCompute(VkCommandBuffer cmd, const VkExtent2D &size)
{
    auto sbdg = m_dutil->DBG_SCOPE(cmd);

    vkCmdPushConstants(cmd, m_dset->getPipeLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GbufConst), &m_pushConst);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dset->getPipeLayout(), 0, static_cast<uint32_t>(m_writes.size()), m_writes.data());
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdDispatch(cmd, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
}

bool GBuffer::onUI()
{
    bool changed = false;
    using namespace ImGuiH;
    PropertyEditor::begin();
    PropertyEditor::end();
    return changed;
}