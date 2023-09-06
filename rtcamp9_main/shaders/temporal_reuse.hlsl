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

int2 getPrevPixel(float3 pos, float2 imgSize)
{

    float4x4 viewproj = mul(frameInfo.prevView, frameInfo.prevProj);
    float4 prevClip = mul(viewproj, float4(pos, 1.0));
    float2 prevUV = float2(((prevClip.x) / (prevClip.w)) * 0.5 + 0.5, ((-prevClip.y) / (prevClip.w)) * 0.5 + 0.5);
    int2 prevPix = int2(prevUV.x * (float)imgSize.x, prevUV.y * (float)imgSize.y);
    return prevPix;
}

void resamplePixel(float3 rayDir, inout uint seed, uint2 pixel, float2 launchSize, uint pixel1d, uint prevPixel1d)
{

    Reservoir centerRes = unpack(packedInReservoir[pixel1d]);
    Reservoir oldRes = unpack(packedTemporalReservoir[prevPixel1d]);

    if (true)
    {
        GBufStruct gbuf = gbuffer1d[pixel1d];
        float3 v1Thp = float3(0.0f, 0.0f, 0.0f);

        // Retrieve the material color
        uint64_t matOffset = sizeof(Material) * gbuf.matId;
        Material mat = getMaterial(sceneDesc[0].materialAddress, matOffset);

        // Setting up the material
        PbrMaterial pbrMat;
        pbrMat.albedo = float4(mat.albedo, 1);
        pbrMat.roughness = mat.roughness;
        pbrMat.metallic = mat.metallic;
        pbrMat.normal = gbuf.nrm;
        pbrMat.emissive = float3(0.0F, 0.0F, 0.0F);
        pbrMat.f0 = lerp(float3(0.04F, 0.04F, 0.04F), pbrMat.albedo.xyz, mat.metallic);

        // Sample BSDF
        {
            BsdfSampleData sampleData;
            sampleData.k1 = -rayDir;
            uint copiedSeed = centerRes.s.seed;
            sampleData.xi = float4(rand(copiedSeed), rand(copiedSeed), rand(copiedSeed), rand(copiedSeed));

            bsdfSample(sampleData, pbrMat);
            if (sampleData.event_type == BSDF_EVENT_ABSORB)
            {
                ;
                // break; // Need to add the contribution ?
            }
            v1Thp = sampleData.bsdf_over_pdf;
        }
        float inv_p = oldRes.w / toScalar(oldRes.s.p_hat_xi);
        Sample Ts;
        {
            Ts.k = oldRes.s.k;
            Ts.seed = oldRes.s.seed;
            Ts.to = oldRes.s.to;
            // Ts.primId = oldRes.s.primId;

            Ts.p_hat_xi = oldRes.s.p_hat_cached * v1Thp; // TODO:
            Ts.p_hat_cached = oldRes.s.p_hat_cached;
        }
        float mi = centerRes.w / (oldRes.wSum + centerRes.w);
        float w = mi * toScalar(centerRes.s.p_hat_xi) * inv_p;

        updateReservoir(oldRes, centerRes.s, w, centerRes.M, rand(seed));
        // updateReservoir(centerRes, oldRes.s, w, 1, rand(seed));
    }
    // if (centerRes.M > SUB_FRAMES)
    // {
    //     centerRes.wSum *= (float)(SUB_FRAMES) / (centerRes.M);
    //     centerRes.M = SUB_FRAMES;
    // }
    packedOutReservoir[pixel1d] = pack(oldRes);

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
    // packedOutReservoir[pixel1d] = packedInReservoir[pixel1d];
    // return;

    if (frameInfo.frame == 0 || !USE_TEMPORAL)
    {
        packedOutReservoir[pixel1d] = packedInReservoir[pixel1d];
        return;
    }

    // // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
    // const float2 subpixel_jitter = frameInfo.frame == 0 ? float2(0.5f, 0.5f) : float2(rand(seed), rand(seed));
    // const float2 pixelCenter = (float2)pixel + subpixel_jitter;
    // const float2 inUV = pixelCenter / (float2)imgSize;
    // const float2 d = inUV * 2.0 - 1.0;
    // const float4 target = mul(frameInfo.projInv, float4(d.x, d.y, 0.01, 1.0));
    // float3 rayDir = mul(frameInfo.viewInv, float4(normalize(target.xyz), 0.0)).xyz;

    // resamplePixel(rayDir, seed, pixel, imgSize, pixel1d, pixel1d);
    // return;

    if (frameInfo.frame % SUB_FRAMES == 0)
    {
        packedOutReservoir[pixel1d] = packedInReservoir[pixel1d];

        // Reservoir empty = initReservoir();

        // const float2 subpixel_jitter = frameInfo.frame == 0 ? float2(0.5f, 0.5f) : float2(rand(seed), rand(seed));
        // const float2 pixelCenter = (float2)pixel + subpixel_jitter;
        // const float2 inUV = pixelCenter / (float2)imgSize;
        // const float2 d = inUV * 2.0 - 1.0;
        // const float4 target = mul(frameInfo.projInv, float4(d.x, d.y, 0.01, 1.0));
        // float3 rayDir = mul(frameInfo.viewInv, float4(normalize(target.xyz), 0.0)).xyz;

        // GBufStruct gbuf = gbuffer1d[pixel1d];
        // int2 prevPixel = getPrevPixel(gbuf.pos, imgSize);
        // uint prevPixel1d = prevPixel.x + imgSize.x * prevPixel.y;
        // GBufStruct maybePrevgbuf = gbuffer1d[prevPixel1d];
        // float distance = length(gbuf.pos - maybePrevgbuf.pos);
        // if (distance < 10.0)
        // {
        //     resamplePixel(rayDir, seed, pixel, imgSize, pixel1d, prevPixel1d);
        // }
        // else
        // {
        //     packedOutReservoir[pixel1d] = packedInReservoir[pixel1d];
        // }
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

        resamplePixel(rayDir, seed, pixel, imgSize, pixel1d, pixel1d);
        // thpOutImage[pixel] = float4(thp, 1.0f);
    }

    return;
}
