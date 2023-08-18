#pragma once
#include <array>
#include <vulkan/vulkan_core.h>
#include "nvvk/buffers_vk.hpp"
#include "shaders/device_host.h"

namespace rtcamp9
{
    struct SceneResources
    {
        struct PrimitiveMeshVk
        {
            nvvk::Buffer vertices;
            nvvk::Buffer indices;
        };
        std::vector<PrimitiveMeshVk> bMeshes;
        nvvk::Buffer bFrameInfo;
        nvvk::Buffer bPrimInfo;
        nvvk::Buffer bSceneDesc;
        nvvk::Buffer bInstInfoBuffer;
        nvvk::Buffer bMaterials;

        std::vector<nvh::PrimitiveMesh> meshes;
        std::vector<nvh::Node> nodes;
        std::vector<Material> materials;
        Light light;
    };
}