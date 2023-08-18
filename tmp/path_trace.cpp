#pragma once
#include "path_trace.hpp"

#if USE_HLSL
#include "_autogen/ray_query_computeMain.spirv.h"
const auto &comp_shd = std::vector<char>{std::begin(ray_query_computeMain), std::end(ray_query_computeMain)};
#elif USE_SLANG
#include "_autogen/ray_query_computeMain.spirv.h"
const auto &comp_shd = std::vector<uint32_t>{std::begin(ray_query_computeMain), std::end(ray_query_computeMain)};
#endif

#define dummy 1

namespace rtcamp9
{
    PathTrace::PathTrace(nvvk::Context *ctx, nvvkhl::AllocVma *alloc)
        : m_ctx(ctx), m_dutil(std::make_unique<nvvk::DebugUtil>(ctx->m_device)), m_alloc(std::make_unique<nvvkhl::AllocVma>(ctx)), m_rtSet(std::make_unique<nvvk::DescriptorSetContainer>(ctx->m_device))
    {
        // Rt props
        VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        prop2.pNext = &m_rtProperties;
        vkGetPhysicalDeviceProperties2(m_ctx->m_physicalDevice, &prop2);

        //
        auto gctQueueIndex = m_ctx->m_queueGCT.familyIndex;
        m_rtBuilder.setup(m_ctx->m_device, m_alloc.get(), gctQueueIndex);
        m_sbt.setup(m_ctx->m_device, gctQueueIndex, m_alloc.get(), m_rtProperties);
    }

    PathTrace::~PathTrace()
    {
        vkDeviceWaitIdle(m_ctx->m_device);
        vkDestroyPipelineLayout(m_ctx->m_device, m_pipelineLayout, nullptr);
        vkDestroyPipeline(m_ctx->m_device, m_graphicsPipeline, nullptr);
        m_rtSet->deinit();
        m_rtPipe.destroy(m_ctx->m_device);
        m_sbt.destroy();
        m_rtBuilder.destroy();
    }

    void PathTrace::updateComputeDescriptorSets()
    {
        VkAccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();
        VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        descASInfo.accelerationStructureCount = 1;
        descASInfo.pAccelerationStructures = &tlas;

        VkDescriptorBufferInfo dbi_unif{m_sceneResources->bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo sceneDesc{m_sceneResources->bSceneDesc.buffer, 0, VK_WHOLE_SIZE};

        m_writes.clear();
        m_writes.emplace_back(m_rtSet->makeWrite(0, B_tlas, &descASInfo));
        // auto asdesc = m_writes[0];
        m_writes.emplace_back(m_rtSet->makeWrite(0, B_outImage, m_oimage.get()));
        m_writes.emplace_back(m_rtSet->makeWrite(0, B_outBuffer, m_outBuffer.get()));
        m_writes.emplace_back(m_rtSet->makeWrite(0, B_frameInfo, &dbi_unif));
        m_writes.emplace_back(m_rtSet->makeWrite(0, B_sceneDesc, &sceneDesc));
    }

    void PathTrace::updateComputeDescriptorSets(VkDescriptorImageInfo outImage)
    {
        m_oimage = std::make_unique<VkDescriptorImageInfo>(outImage);
        // m_outBuffer = std::make_unique<VkDescriptorBufferInfo>(outBuffer);
        VkAccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();
        VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        descASInfo.accelerationStructureCount = 1;
        descASInfo.pAccelerationStructures = &tlas;

        VkDescriptorBufferInfo dbi_unif{m_sceneResources->bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo sceneDesc{m_sceneResources->bSceneDesc.buffer, 0, VK_WHOLE_SIZE};

        m_writes.clear();
        m_writes.emplace_back(m_rtSet->makeWrite(0, B_tlas, &descASInfo));
        m_writes.emplace_back(m_rtSet->makeWrite(0, B_outImage, m_oimage.get()));
        m_writes.emplace_back(m_rtSet->makeWrite(0, B_outBuffer, m_outBuffer.get()));
        m_writes.emplace_back(m_rtSet->makeWrite(0, B_frameInfo, &dbi_unif));
        m_writes.emplace_back(m_rtSet->makeWrite(0, B_sceneDesc, &sceneDesc));
    }

    VkCommandBuffer PathTrace::prepareResources(VkCommandBuffer cmd, const VkExtent2D &size)
    {
        RadianceInfo data{};
        data.radiance = vec3(0.0f);
        std::vector<RadianceInfo>
            r_info(size.height * size.width, data);
        auto buffer = m_alloc->createBuffer(cmd, r_info, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        m_dutil->DBG_NAME(buffer.buffer);

        VkDescriptorBufferInfo bufInfo{buffer.buffer, 0, VK_WHOLE_SIZE};
        m_outBuffer = std::make_unique<VkDescriptorBufferInfo>(bufInfo);

        return cmd;
    }

    void PathTrace::runCompute(VkCommandBuffer cmd, const VkExtent2D &size)
    {

        auto sbdg = m_dutil->DBG_SCOPE(cmd);

        float view_aspect_ratio = size.height / size.width;
        nvmath::vec3f eye;
        nvmath::vec3f center;
        nvmath::vec3f up;
        CameraManip.getLookat(eye, center, up);

        const auto &clip = CameraManip.getClipPlanes();
        FrameInfo finfo{
            .proj = nvmath::perspectiveVK(CameraManip.getFov(), view_aspect_ratio, clip.x, clip.y),
            .view = CameraManip.getMatrix(),
            .projInv = nvmath::inverse(finfo.proj),
            .viewInv = nvmath::inverse(finfo.view),
            .prevProj = m_finfo.proj,
            .prevView = m_finfo.view,
            .prevProjInv = m_finfo.projInv,
            .prevViewInv = m_finfo.viewInv,
            .camPos = eye,
        };
        vkCmdUpdateBuffer(cmd, m_sceneResources->bFrameInfo.buffer, 0, sizeof(FrameInfo), &finfo);
        m_finfo = finfo;
        m_pushConst.frame = m_frame;
        m_pushConst.light = m_sceneResources->light;

        VkMemoryBarrier memBarrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

        std::vector<VkDescriptorSet> descSets{m_rtSet->getSet()};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipe.plines[0]);
        pushDescriptorSet(cmd);
        // updateComputeDescriptorSets();
        // vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipe.layout, 0,
        //                           static_cast<uint32_t>(m_writes.size()), m_writes.data());

        vkCmdPushConstants(cmd, m_rtPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant), &m_pushConst);

        vkCmdDispatch(cmd, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);
        // auto buf_memory_barrier = ;
    }

    void PathTrace::pushDescriptorSet(VkCommandBuffer cmd)
    {
        // Write to descriptors
        VkAccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();
        VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        descASInfo.accelerationStructureCount = 1;
        descASInfo.pAccelerationStructures = &tlas;
        // VkDescriptorImageInfo imageInfo{{}, m_gBuffers->getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorBufferInfo dbi_unif{m_sceneResources->bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo sceneDesc{m_sceneResources->bSceneDesc.buffer, 0, VK_WHOLE_SIZE};

        std::vector<VkWriteDescriptorSet> writes;
        writes.emplace_back(m_rtSet->makeWrite(0, B_tlas, &descASInfo));
        writes.emplace_back(m_rtSet->makeWrite(0, B_outImage, m_oimage.get()));
        writes.emplace_back(m_rtSet->makeWrite(0, B_outBuffer, m_outBuffer.get()));
        writes.emplace_back(m_rtSet->makeWrite(0, B_frameInfo, &dbi_unif));
        writes.emplace_back(m_rtSet->makeWrite(0, B_sceneDesc, &sceneDesc));

        vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipe.layout, 0,
                                  static_cast<uint32_t>(writes.size()), writes.data());
    }

    void PathTrace::setScene(const std::shared_ptr<SceneResources> &sceneResources)
    {
        m_sceneResources = sceneResources;
        m_pushConst.maxDepth = 5;
        m_pushConst.frame = 0;
        m_pushConst.fireflyClampThreshold = 10;
        m_pushConst.maxSamples = 1;
        m_pushConst.light = m_sceneResources->light;
        float a = 1.0f;

        createBottomLevelAS();
        createTopLevelAS();
    }

    nvvk::RaytracingBuilderKHR::BlasInput PathTrace::primitiveToGeometry(const nvh::PrimitiveMesh &prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress)
    {
        uint32_t maxPrimitiveCount = prim.triangles.size();

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

    void PathTrace::createBottomLevelAS()
    {
        std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
        allBlas.reserve(m_sceneResources->meshes.size());

        for (size_t i = 0; i < m_sceneResources->meshes.size(); i++)
        {
            auto vertexAddress = nvvk::getBufferDeviceAddress(m_ctx->m_device, m_sceneResources->bMeshes[i].vertices.buffer);
            auto indexAddress = nvvk::getBufferDeviceAddress(m_ctx->m_device, m_sceneResources->bMeshes[i].indices.buffer);

            auto geo = primitiveToGeometry(m_sceneResources->meshes[i], vertexAddress, indexAddress);
            allBlas.push_back({geo});
        }
        m_rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
    }

    void PathTrace::createTopLevelAS()
    {
        std::vector<VkAccelerationStructureInstanceKHR> tlas;
        tlas.reserve(m_sceneResources->nodes.size());
        for (auto &node : m_sceneResources->nodes)
        {
            VkGeometryInstanceFlagsKHR flags{VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV};

            VkAccelerationStructureInstanceKHR rayInst{};
            rayInst.transform = nvvk::toTransformMatrixKHR(node.localMatrix()); // Position of the instance
            rayInst.instanceCustomIndex = node.mesh;                            // gl_InstanceCustomIndexEXT
            rayInst.accelerationStructureReference = m_rtBuilder.getBlasDeviceAddress(node.mesh);
            rayInst.instanceShaderBindingTableRecordOffset = 0; // We will use the same hit group for all objects
            rayInst.flags = flags;
            rayInst.mask = 0xFF;
            tlas.emplace_back(rayInst);
        }
        auto flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        m_rtBuilder.buildTlas(tlas, flags);
    }

    void PathTrace::createComputePipeline()
    {

        m_rtPipe.destroy(m_ctx->m_device);
        m_rtSet->deinit();
        m_rtSet = std::make_unique<nvvk::DescriptorSetContainer>(m_ctx->m_device);
        m_rtPipe.plines.resize(1);

        m_rtSet->addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
        m_rtSet->addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
        m_rtSet->addBinding(B_outBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
        m_rtSet->addBinding(B_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
        m_rtSet->addBinding(B_sceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
        m_rtSet->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

        VkPushConstantRange pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant)};
        VkPipelineLayoutCreateInfo plCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &m_rtSet->getLayout(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstant,
        };
        vkCreatePipelineLayout(m_ctx->m_device, &plCreateInfo, nullptr, &m_rtPipe.layout);
        VkComputePipelineCreateInfo cpCreateInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = nvvk::createShaderStageInfo(m_ctx->m_device, comp_shd, VK_SHADER_STAGE_COMPUTE_BIT, "computeMain"),
            .layout = m_rtPipe.layout,
        };
        vkCreateComputePipelines(m_ctx->m_device, {}, 1, &cpCreateInfo, nullptr, &m_rtPipe.plines[0]);
        vkDestroyShaderModule(m_ctx->m_device, cpCreateInfo.stage.module, nullptr);
    }

    void PathTrace::resetFrame()
    {
        m_frame = -1;
        m_frame = -1;
    }

    bool PathTrace::updateFrame()
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
        return true;
    }

    bool PathTrace::onUI()
    {
        bool changed{false};
        using namespace ImGuiH;
        PropertyEditor::begin();
        if (PropertyEditor::treeNode("Light"))
        {
            changed |= PropertyEditor::entry("Position", [&]
                                             { return ImGui::DragFloat3("#1", &m_sceneResources->light.position.x); });

            changed |= PropertyEditor::entry("Intensity", [&]
                                             { return ImGui::SliderFloat("#1", &m_sceneResources->light.intensity, 0.0F, 1000.0F, "%.3f", ImGuiSliderFlags_Logarithmic); });
            changed |=
                PropertyEditor::entry("Radius", [&]
                                      { return ImGui::SliderFloat("#1", &m_sceneResources->light.radius, 0.0F, 1.0F); });
            PropertyEditor::treePop();
        }
        if (PropertyEditor::treeNode("Ray Tracer"))
        {
            changed |= PropertyEditor::entry("Depth", [&]
                                             { return ImGui::SliderInt("#1", &m_pushConst.maxDepth, 0, 20); });
            changed |=
                PropertyEditor::entry("Samples", [&]
                                      { return ImGui::SliderInt("#1", &m_pushConst.maxSamples, 1, 10); });
            PropertyEditor::treePop();
        }
        PropertyEditor::end();
        return changed;
    }
}