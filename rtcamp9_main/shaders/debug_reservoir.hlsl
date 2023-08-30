#include "dh_vis_binding.h"

#include "constants.hlsli"
#include "device_host.h"
#include "dh_reservoir.hlsl"
#include "reservoir.hlsl"

[[vk::push_constant]]
ConstantBuffer<PushConstant> pushConst;
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
    float3 outColor;

    // float3 rad = thpInImage[pixel].xyz * centerRes.s.p_hat_xi * calcContributionWegiht(centerRes);
    // gOutImage[pixel] = float4(thpInImage[pixel].xyz * centerRes.s.p_hat_cached * calcContributionWegiht(centerRes), 1.0f);

    bool reuseOk = centerRes.s.k == 1;
    if (!reuseOk)
    {
        outColor = gRes[pixel1D].radiance;
        // outColor /= pushConst.maxSamples;
        // outColor = centerRes.s.p_hat_xi;
    }
    else
    {
        float3 reconstructRad = thpInImage[pixel].xyz * centerRes.s.p_hat_cached * calcContributionWegiht(centerRes);
        // float3 reconstructRad = centerRes.s.p_hat_xi * calcContributionWegiht(centerRes);
        outColor = reconstructRad;
    }

    float lum = dot(outColor, float3(0.212671F, 0.715160F, 0.072169F));
    if (lum > pushConst.fireflyClampThreshold)
    {
        outColor *= pushConst.fireflyClampThreshold / lum;
    }

    bool first_frame = (pushConst.frame == 0);
    if (first_frame)
    {
        gOutImage[pixel] = float4(outColor, 1.0f);
    }
    else
    {
        int subframe = pushConst.frame % SUB_FRAMES;
        float a = 1.0f / float(subframe + 1);

        float3 old_color = gOutImage[pixel].xyz;
        outColor = lerp(old_color, outColor, a);
        gOutImage[pixel] = float4(outColor, 1.0f);
        // gOutImage[pixel] = float4(a, a, a, 1.0f);
    }
}
