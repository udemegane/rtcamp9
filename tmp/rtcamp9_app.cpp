#pragma once
#define VMA_IMPLEMENTATION
#include "path_trace.hpp"
#include "scene_resources.hpp"
using namespace rtcamp9;

class PathResampling : public nvvkhl::IAppElement
{
    enum struct ETraceMode
    {
        NaivePathtrace,
    };

public:
    PathResampling() = default;
    ~PathResampling() override = default;

    void onAttach(nvvkhl::Application *app) override
    {
        m_app = app;
        m_device = m_app->getDevice();
        m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);
        m_alloc = std::make_shared<nvvkhl::AllocVma>(m_app->getContext().get());
        m_pathtracer = std::make_unique<PathTrace>(m_app->getContext().get(), m_alloc.get());

        // TODO
        createScene();
        createVkBuffers();
        m_pathtracer->setScene(m_sceneResources);
        m_pathtracer->createComputePipeline();
    }

    void onDetach() override
    {
        vkDeviceWaitIdle(m_device);
        destroyResources();
    }

    void onResize(uint32_t width, uint32_t height) override
    {
        createGbuffers({width, height});
        {
            auto *cmd = m_app->createTempCmdBuffer();
            auto size = m_app->getViewportSize();
            m_pathtracer->prepareResources(cmd, m_app->getViewportSize());
            m_app->submitAndWaitTempCmdBuffer(cmd);
        }
        m_pathtracer->updateComputeDescriptorSets(m_gBuffers->getDescriptorImageInfo(gbufferOutIdx));
        m_pathtracer->resetFrame();
        // m_pathtracer->
    }

    void onUIRender() override
    {
        {
            ImGui::Begin("Settings");
            ImGuiH::CameraWidget();
            using namespace ImGuiH;
            bool changed{false};

            if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
            {
                changed |= m_pathtracer->onUI();
            }
            ImGui::End();
        }

        { // Rendering Viewport
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
            ImGui::Begin("Viewport");

            // Display the G-Buffer image
            ImGui::Image(m_gBuffers->getDescriptorSet(0), ImGui::GetContentRegionAvail());

            ImGui::End();
            ImGui::PopStyleVar();
        }
    }

    void onRender(VkCommandBuffer cmd) override
    {

        m_pathtracer->runCompute(cmd, m_app->getViewportSize());
        // Making sure the rendered image is ready to be used
        auto image_memory_barrier =
            nvvk::makeImageMemoryBarrier(m_gBuffers->getColorImage(1), VK_ACCESS_SHADER_READ_BIT,
                                         VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &image_memory_barrier);
    }

private:
    void createScene()
    {
        m_sceneResources = std::make_shared<SceneResources>();
        m_sceneResources->materials.push_back({{0.985f, 0.862f, 0.405f}, 0.5f, 0.0f});
        m_sceneResources->materials.push_back({{0.622f, 0.928f, 0.728f}, 0.05f, 1.0f});
        m_sceneResources->materials.push_back({{.7F, .7F, .7F}, 0.3f, 0.0f});

        m_sceneResources->meshes.emplace_back(nvh::createCube(1, 1, 1));
        m_sceneResources->meshes.emplace_back(nvh::createSphereUv(0.5f));
        m_sceneResources->meshes.emplace_back(nvh::createPlane(10, 100, 100));

        // Instance Cube
        {
            auto &n = m_sceneResources->nodes.emplace_back();
            n.mesh = 0;
            n.material = 0;
            n.translation = {0.0f, 0.5f, 0.0F};
        }

        // Instance Sphere
        {
            auto &n = m_sceneResources->nodes.emplace_back();
            n.mesh = 1;
            n.material = 1;
            n.translation = {1.0f, 1.5f, 1.0F};
        }

        // Adding a plane & material
        {
            auto &n = m_sceneResources->nodes.emplace_back();
            n.mesh = 2;
            n.material = 2;
            n.translation = {0.0f, 0.0f, 0.0f};
        }

        m_sceneResources->light.intensity = 100.0f;
        m_sceneResources->light.position = {2.0f, 7.0f, 2.0f};
        m_sceneResources->light.radius = 0.2f;

        // Setting camera to see the scene
        CameraManip.setClipPlanes({0.1F, 100.0F});
        CameraManip.setLookat({-2.0F, 2.5F, 3.0f}, {0.4F, 0.3F, 0.2F}, {0.0F, 1.0F, 0.0F});
    }

    void createVkBuffers()
    {
        auto *cmd = m_app->createTempCmdBuffer();
        m_sceneResources->bMeshes.resize(m_sceneResources->meshes.size());

        auto rtUsageFlag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

        //
        std::vector<PrimMeshInfo> primInfo;
        for (size_t i = 0; i < m_sceneResources->meshes.size(); i++)
        {
            auto &m = m_sceneResources->bMeshes[i];
            m.vertices = m_alloc->createBuffer(cmd, m_sceneResources->meshes[i].vertices, rtUsageFlag);
            m.indices = m_alloc->createBuffer(cmd, m_sceneResources->meshes[i].triangles, rtUsageFlag);
            m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
            m_dutil->DBG_NAME_IDX(m.indices.buffer, i);

            PrimMeshInfo info{
                .vertexAddress = nvvk::getBufferDeviceAddress(m_device, m.vertices.buffer),
                .indexAddress = nvvk::getBufferDeviceAddress(m_device, m.indices.buffer),
            };
            primInfo.emplace_back(info);
        }
        m_sceneResources->bPrimInfo = m_alloc->createBuffer(cmd, primInfo, rtUsageFlag);
        m_dutil->DBG_NAME(m_sceneResources->bPrimInfo.buffer);

        m_sceneResources->bFrameInfo = m_alloc->createBuffer(sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        m_dutil->DBG_NAME(m_sceneResources->bFrameInfo.buffer);

        std::vector<InstanceInfo> instInfo;
        for (auto &node : m_sceneResources->nodes)
        {
            InstanceInfo info{
                info.transform = node.localMatrix(),
                info.materialID = node.material,
            };
            instInfo.emplace_back(info);
        }
        m_sceneResources->bInstInfoBuffer =
            m_alloc->createBuffer(cmd, instInfo, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        m_dutil->DBG_NAME(m_sceneResources->bInstInfoBuffer.buffer);

        // Buffer references of all scene elements
        SceneDescription sceneDesc{
            .materialAddress = nvvk::getBufferDeviceAddress(m_device, m_sceneResources->bMaterials.buffer),
            .instInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_sceneResources->bInstInfoBuffer.buffer),
            .primInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_sceneResources->bPrimInfo.buffer),
            .light = m_sceneResources->light,
        };

        m_sceneResources->bSceneDesc = m_alloc->createBuffer(cmd, sizeof(SceneDescription), &sceneDesc,
                                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        m_dutil->DBG_NAME(m_sceneResources->bSceneDesc.buffer);

        m_app->submitAndWaitTempCmdBuffer(cmd);
    }

    void createGbuffers(const nvmath::vec2f &size)
    {
        m_viewSize = size;
        std::vector<VkFormat> color_buffers = {m_colorFormat, m_colorFormat}; // tonemapped, original
        m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(),
                                                       VkExtent2D{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)},
                                                       color_buffers, m_depthFormat);
    }

    void destroyResources()
    {
        for (auto &m : m_sceneResources->bMeshes)
        {
            m_alloc->destroy(m.vertices);
            m_alloc->destroy(m.indices);
        }
        m_alloc->destroy(m_sceneResources->bFrameInfo);
        m_alloc->destroy(m_sceneResources->bPrimInfo);
        m_alloc->destroy(m_sceneResources->bSceneDesc);
        m_alloc->destroy(m_sceneResources->bInstInfoBuffer);
        m_alloc->destroy(m_sceneResources->bMaterials);

        m_gBuffers.reset();
        m_pathtracer.reset();
    }

    nvvkhl::Application *m_app{nullptr};
    std::unique_ptr<nvvk::DebugUtil> m_dutil;
    std::unique_ptr<PathTrace> m_pathtracer;
    std::shared_ptr<nvvkhl::AllocVma> m_alloc;

    nvmath::vec2f m_viewSize = {1, 1};
    VkFormat m_colorFormat = VK_FORMAT_R32G32B32A32_SFLOAT; // Color format of the image
    VkFormat m_depthFormat = VK_FORMAT_D32_SFLOAT;          // VK_FORMAT_X8_D24_UNORM_PACK32// Depth format of the depth buffer
    VkDevice m_device = VK_NULL_HANDLE;                     // Convenient
    std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;            // G-Buffers: color + depth
    static const int gbufferOutIdx = 0;

    std::shared_ptr<SceneResources> m_sceneResources;
};

int main(int argc, char **argv)
{
    nvvkhl::ApplicationCreateInfo spec;
    spec.name = PROJECT_NAME " RTCamp9";
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

    // Create the application
    auto app = std::make_unique<nvvkhl::Application>(spec);

    // Create the test framework
    auto test = std::make_shared<nvvkhl::ElementTesting>(argc, argv);

    // Add all application elements
    app->addElement(test);
    app->addElement(std::make_shared<nvvkhl::ElementCamera>());
    app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>()); // Menu / Quit
    app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>());
    app->addElement(std::make_shared<PathResampling>());

    app->run();
    app.reset();

    return test->errorCode();
}