#pragma once
#include "device_host.h"
#include "dh_gbuf.h"
#include "dh_reservoir.hlsl"
#include "dh_reuse.h"

#include "constants.hlsli"
#include "ggx.hlsli"
#include "random.hlsli"
#include "reservoir.hlsl"

[[vk::binding(B_reuse_frameinfo)]]
ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(B_reuse_gbuffer)]]
StructuredBuffer<GBufStruct> gbuffer1d;
[[vk::binding(B_reuse_scenedesc)]]
StructuredBuffer<SceneDescription> sceneDesc;
[[vk::binding(B_reuse_inReservoir)]]
StructuredBuffer<PackedReservoir> packedInReservoir;
[[vk::binding(B_reuse_oldReservoir)]]
StructuredBuffer<PackedReservoir> packedTemporalReservoir;
[[vk::binding(B_reuse_outReservoir)]]
RWStructuredBuffer<PackedReservoir> packedOutReservoir;
[[vk::binding(B_reuse_outThp)]]
RWTexture2D<float4> thpOutImage;

Material getMaterial(uint64_t materialAddress, uint64_t offset)
{
    Material m;
    m.albedo = vk::RawBufferLoad<float3>(materialAddress + offset);
    m.roughness = vk::RawBufferLoad<float>(materialAddress + offset + sizeof(float3));
    m.metallic = vk::RawBufferLoad<float>(materialAddress + offset + sizeof(float3) + sizeof(float));
    return m;
}

void resamplePixel(float3 rayDir, inout uint seed, uint2 pixel, float2 launchSize, uint pixel1d)
{
    Reservoir centerRes = unpack(packedInReservoir[pixel1d]);
    Reservoir oldRes = unpack(packedTemporalReservoir[pixel1d]);

    float inv_p = oldRes.w / toScalar(oldRes.s.p_hat_xi);
    // Sample Ts;
    // {
    //     Ts.k = ri.s.k;
    //     Ts.seed = ri.s.seed;
    //     Ts.to = ri.s.to;
    //     Ts.primId = ri.s.primId;

    //     Ts.p_hat_xi = ri.s.p_hat_cached * v1Thp; // TODO:
    //     Ts.p_hat_cached = ri.s.p_hat_cached;
    // }
    float w = toScalar(oldRes.s.p_hat_xi) * inv_p;

    updateReservoir(centerRes, oldRes.s, w, 1, rand(seed));

    // GBufStruct gbuf = gbuffer1d[pixel1d];
    // float3 v1Thp = float3(0.0f, 0.0f, 0.0f);

    // // Retrieve the material color
    // uint64_t matOffset = sizeof(Material) * gbuf.matId;
    // Material mat = getMaterial(sceneDesc[0].materialAddress, matOffset);

    // // Setting up the material
    // PbrMaterial pbrMat;
    // pbrMat.albedo = float4(mat.albedo, 1);
    // pbrMat.roughness = mat.roughness;
    // pbrMat.metallic = mat.metallic;
    // pbrMat.normal = gbuf.nrm;
    // pbrMat.emissive = float3(0.0F, 0.0F, 0.0F);
    // pbrMat.f0 = lerp(float3(0.04F, 0.04F, 0.04F), pbrMat.albedo.xyz, mat.metallic);

    // // Sample BSDF
    // {
    //     BsdfSampleData sampleData;
    //     sampleData.k1 = -rayDir;
    //     uint copiedSeed = centerRes.s.seed;
    //     sampleData.xi = float4(rand(copiedSeed), rand(copiedSeed), rand(copiedSeed), rand(copiedSeed));

    //     bsdfSample(sampleData, pbrMat);
    //     if (sampleData.event_type == BSDF_EVENT_ABSORB)
    //     {
    //         ;
    //         // break; // Need to add the contribution ?
    //     }
    //     v1Thp = sampleData.bsdf_over_pdf;
    // }

    packedOutReservoir[pixel1d] = pack(centerRes);
    // return v1Thp;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID, uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    uint2 imgSize;
    imgSize.x = frameInfo.width;
    imgSize.y = frameInfo.height;
    if (pixel.x >= imgSize.x || pixel.y >= imgSize.y)
        return;
    uint pixel1d = pixel.x + imgSize.x * pixel.y;
    // Initialize the random number
    uint seed = xxhash32(uint3(pixel.xy, frameInfo.frame));

    if (frameInfo.frame % SUB_FRAMES == 0)
    {
        // Reservoir empty = initReservoir();
        packedOutReservoir[pixel1d] = packedInReservoir[pixel1d];
    }
    else
    {
        // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
        const float2 subpixel_jitter = frameInfo.frame == 0 ? float2(0.5f, 0.5f) : float2(rand(seed), rand(seed));
        const float2 pixelCenter = (float2)pixel + subpixel_jitter;
        const float2 inUV = pixelCenter / (float2)imgSize;
        const float2 d = inUV * 2.0 - 1.0;
        const float4 target = mul(frameInfo.projInv, float4(d.x, d.y, 0.01, 1.0));
        float3 rayDir = mul(frameInfo.viewInv, float4(normalize(target.xyz), 0.0)).xyz;

        resamplePixel(rayDir, seed, pixel, imgSize, pixel1d);
        // thpOutImage[pixel] = float4(thp, 1.0f);
    }

    return;
}
