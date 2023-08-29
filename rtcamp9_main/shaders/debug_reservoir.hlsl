#include "dh_vis_binding.h"

#include "constants.hlsli"
#include "dh_reservoir.hlsl"
#include "reservoir.hlsl"

[[vk::push_constant]]
ConstantBuffer<DBGConstant> pushConst;
[[vk::binding(eDebugPassInput)]]
RWStructuredBuffer<DIReservoir> gRes;
[[vk::binding(B_compose_giInput)]]
StructuredBuffer<PackedReservoir> packedGiRes;
[[vk::binding(B_compose_thpInput)]]
Texture2D<float4> thpInImage;
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
    Reservoir centerRes = unpack(packedGiRes[pixel1D]);
    float3 rad = thpInImage[pixel].xyz * centerRes.s.radiance * calcContributionWegiht(centerRes);

    if (/*pixel.x >  imgSize.x / 2*/ true)
    {
        gOutImage[pixel] = float4(gRes[pixel1D].radiance, 1.0f);
    }
    else
    {
        // thp違う
        //
        // gOutImage[pixel] = thpInImage[pixel];
        // radiance も違う
        // gOutImage[pixel] = float4(centerRes.s.radiance, 1.0f);
        float tmp = calcContributionWegiht(centerRes);
        gOutImage[pixel] = float4(tmp, tmp, tmp, 1.0f);
    }
    // gOutImage[pixel] = float4(thpInImage[pixel].xyz, 1.0f);
}
