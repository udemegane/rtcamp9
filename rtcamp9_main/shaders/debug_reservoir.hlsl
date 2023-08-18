#include "dh_vis_binding.h"

#include "DIReservoir.hlsl"
#include "constants.hlsli"

[[vk::push_constant]]
ConstantBuffer<DBGConstant> pushConst;
[[vk::binding(eDebugPassInput)]]
RWStructuredBuffer<Reservoir> gRes;
[[vk::binding(eDebugPassOutput)]]
RWTexture2D<float4> gOutImage;

#define dummy
[shader("compute")]
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void main(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID, uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    uint2 imgSize;
    gOutImage.GetDimensions(imgSize.x, imgSize.y); // DispatchRaysDimensions();
    imgSize;

    if (pixel.x >= imgSize.x || pixel.y >= imgSize.y)
        return;
    uint pixel1D = pixel.x + imgSize.x * pixel.y;
    float a;
    gOutImage[pixel] = float4(gRes[pixel1D].radiance, 1.0f);
}
