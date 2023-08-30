#include "dh_vis_binding.h"

#include "constants.hlsli"
#include "device_host.h"
#include "dh_reservoir.hlsl"
#include "reservoir.hlsl"

// [[vk::push_constant]]
// ConstantBuffer<PushConstant> pushConst;
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
    Reservoir centerRes = unpack(packedGiRes[pixel1D]);
    // float3 rad = thpInImage[pixel].xyz * centerRes.s.p_hat_xi * calcContributionWegiht(centerRes);
    // gOutImage[pixel] = float4(thpInImage[pixel].xyz * centerRes.s.p_hat_cached * calcContributionWegiht(centerRes), 1.0f);

    bool reuseOk = centerRes.s.k == 1;
    // float k = centerRes.s.k;
    // float3 tmp = float3(k / 5, k / 5, k / 5);
    // gOutImage[pixel] = float4(tmp, 1.0f);
    // return;
    float3 outColor;
    if (true)
    {
        outColor = gRes[pixel1D].radiance;
        // outColor /= pushConst.maxSamples;
        // outColor = centerRes.s.p_hat_xi;
    }
    else
    {
        float3 reconstructRad = thpInImage[pixel].xyz * centerRes.s.p_hat_cached * calcContributionWegiht(centerRes);
        outColor = reconstructRad;
    }

    // float lum = dot(outColor, float3(0.212671F, 0.715160F, 0.072169F));
    // if (lum > pushConst.fireflyClampThreshold)
    // {
    //     outColor *= pushConst.fireflyClampThreshold / lum;
    // }
    gOutImage[pixel] = float4(outColor, 1.0f);
}
