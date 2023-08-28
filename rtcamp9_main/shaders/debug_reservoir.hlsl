#include "dh_vis_binding.h"

#include "constants.hlsli"
#include "dh_reservoir.hlsl"

[[vk::push_constant]]
ConstantBuffer<DBGConstant> pushConst;
[[vk::binding(eDebugPassInput)]]
RWStructuredBuffer<DIReservoir> gRes;
[[vk::binding(eDebugPassOutput)]]
RWTexture2D<float4> gOutImage;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID, uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    uint2 imgSize;
    gOutImage.GetDimensions(imgSize.x, imgSize.y); // DispatchRaysDimensions();

    if (pixel.x >= imgSize.x || pixel.y >= imgSize.y)
        return;
    uint pixel1D = pixel.x + imgSize.x * pixel.y;
    gOutImage[pixel] = float4(gRes[pixel1D].radiance, 1.0f);
}
