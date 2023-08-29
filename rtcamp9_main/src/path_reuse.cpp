#pragma once
#include "path_reuse.hpp"

static const auto spatialFileName = L"spatial_reuse.hlsl";
static const auto temporalFileName = L"temporal_reuse.hlsl";

PathReuse::PathReuse(nvvk::Context *ctx, nvvkhl::AllocVma *alloc, HLSLShaderCompiler *compiler, EResampleType type)
    : m_ctx(ctx), m_dutil(std::make_unique<nvvk::DebugUtil>(ctx->m_device)), m_dset(std::make_unique<nvvk::DescriptorSetContainer>(ctx->m_device)), m_compiler(compiler), m_type(type)
{
    m_pushConst = GbufConst{};
}

PathReuse::~PathReuse()
{
}

void PathReuse::createPipelineLayout()
{
    nvvk::DebugUtil dbg(m_ctx->m_device);
    m_dset->addBinding(B_reuse_inReservoir, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_dset->addBinding(B_reuse_gbuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_dset->addBinding(B_reuse_frameinfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dset->addBinding(B_reuse_outReservoir, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dset->addBinding(B_reuse_scenedesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dset->addBinding(B_reuse_outThp, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);

    m_dset->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

    // // pushing time
    VkPushConstantRange pushConstant{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GbufConst)};
    m_dset->initPipeLayout(1, &pushConstant);
    m_dutil->DBG_NAME(m_dset->getPipeLayout());
}

void PathReuse::createComputePipeline()
{

    ATL::CComPtr<IDxcBlob> spirvCode;
    switch (m_type)
    {
    case EResampleType::Temporal:
        spirvCode = m_compiler->compile(temporalFileName, L"main");
        /* code */
        break;
    case EResampleType::Spatial:
        spirvCode = m_compiler->compile(spatialFileName, L"main");
        /* code */
        break;
    default:
        return;
        break;
    }
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

void PathReuse::updateComputeDescriptorSets(VkDescriptorBufferInfo inReservoir, VkDescriptorBufferInfo outReservoir, VkDescriptorBufferInfo gbuffer, VkDescriptorImageInfo thpImage, VkDescriptorBufferInfo frameInfo, VkDescriptorBufferInfo sceneInfo)
{
    m_ireservoir = std::make_unique<VkDescriptorBufferInfo>(inReservoir);
    m_gbuffer = std::make_unique<VkDescriptorBufferInfo>(gbuffer);
    m_frameinfo = std::make_unique<VkDescriptorBufferInfo>(frameInfo);
    m_sceneinfo = std::make_unique<VkDescriptorBufferInfo>(sceneInfo);
    m_oreservoir = std::make_unique<VkDescriptorBufferInfo>(outReservoir);
    m_thpimage = std::make_unique<VkDescriptorImageInfo>(thpImage);
    m_writes.clear();
    m_writes.emplace_back(m_dset->makeWrite(0, B_reuse_inReservoir, m_ireservoir.get()));
    m_writes.emplace_back(m_dset->makeWrite(0, B_reuse_outReservoir, m_oreservoir.get()));
    m_writes.emplace_back(m_dset->makeWrite(0, B_reuse_gbuffer, m_gbuffer.get()));
    m_writes.emplace_back(m_dset->makeWrite(0, B_reuse_frameinfo, m_frameinfo.get()));
    m_writes.emplace_back(m_dset->makeWrite(0, B_reuse_scenedesc, m_sceneinfo.get()));
    m_writes.emplace_back(m_dset->makeWrite(0, B_reuse_outThp, m_thpimage.get()));
}

void PathReuse::updateConstants(const VkExtent2D &size)
{
    // m_pushConst.height = size.height;
    // m_pushConst.width = size.width;
}

void PathReuse::runCompute(VkCommandBuffer cmd, const VkExtent2D &size)
{
    auto sbdg = m_dutil->DBG_SCOPE(cmd);

    // vkCmdPushConstants(cmd, m_dset->getPipeLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GbufConst), &m_pushConst);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dset->getPipeLayout(), 0, static_cast<uint32_t>(m_writes.size()), m_writes.data());
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdDispatch(cmd, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
}

bool PathReuse::onUI()
{
    bool changed = false;
    using namespace ImGuiH;
    PropertyEditor::begin();
    PropertyEditor::end();
    return changed;
}
// PathReuse::
