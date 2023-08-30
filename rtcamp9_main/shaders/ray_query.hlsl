/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

// https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#rayquery

#include "device_host.h"
#include "dh_bindings.h"
#include "dh_gbuf.h"
#include "dh_reservoir.hlsl"

#include "constants.hlsli"
#include "ggx.hlsli"
#include "random.hlsli"
#include "reservoir.hlsl"
#include "sky.hlsli"

#define WORKGROUP_SIZE 16

#ifndef mat4
#define mat4 float4x4
#endif
#ifndef vec4
#define vec4 float4
#endif
#ifndef vec3
#define vec3 float3
#endif
#ifndef vec2
#define vec2 float2
#endif

// Bindings
[[vk::constant_id(0)]]
const int USE_SER = 0;
[[vk::push_constant]]
ConstantBuffer<PushConstant> pushConst;
[[vk::binding(B_tlas)]]
RaytracingAccelerationStructure topLevelAS;
[[vk::binding(B_outImage)]]
RWTexture2D<float4> outImage;
[[vk::binding(B_frameInfo)]]
ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(B_sceneDesc)]]
StructuredBuffer<SceneDescription> sceneDesc;
[[vk::binding(B_outBuffer)]]
RWStructuredBuffer<DIReservoir> diReservoir;
[[vk::binding(B_gbuffer)]]
RWStructuredBuffer<GBufStruct> gbuffer1d;
[[vk::binding(B_outReservoir)]]
RWStructuredBuffer<PackedReservoir> packedReservoir;

#define IDENTITY_MATRIX float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)

float4x4 inverse(float4x4 m)
{
    float n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
    float n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
    float n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
    float n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

    float t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
    float t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
    float t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
    float t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

    float det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    float idet = 1.0f / det;

    float4x4 ret;

    ret[0][0] = t11 * idet;
    ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
    ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
    ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

    ret[1][0] = t12 * idet;
    ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
    ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
    ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

    ret[2][0] = t13 * idet;
    ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
    ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
    ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

    ret[3][0] = t14 * idet;
    ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
    ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
    ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;

    return ret;
}

//-----------------------------------------------------------------------
// Payload
// See: https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#example
struct HitPayload
{
    float hitT;
    int instanceIndex;
    float3 pos;
    float3 nrm;
    float3 geonrm;
};

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
    float3 pos;
    float3 nrm;
    float3 geonrm;
};

// Return the Vertex structure, from a buffer address and an offset
Vertex getVertex(uint64_t vertAddress, uint64_t offset)
{
    Vertex v;
    v.position = vk::RawBufferLoad<float3>(vertAddress + offset);
    v.normal = vk::RawBufferLoad<float3>(vertAddress + offset + sizeof(float3));
    v.t = vk::RawBufferLoad<float2>(vertAddress + offset + (2 * sizeof(float3)));
    return v;
}

Material getMaterial(uint64_t materialAddress, uint64_t offset)
{
    Material m;
    m.albedo = vk::RawBufferLoad<float3>(materialAddress + offset);
    m.roughness = vk::RawBufferLoad<float>(materialAddress + offset + sizeof(float3));
    m.metallic = vk::RawBufferLoad<float>(materialAddress + offset + sizeof(float3) + sizeof(float));
    return m;
}

//-----------------------------------------------------------------------
// Return hit position, normal and geometric normal in world space
HitState getHitState(float2 barycentricCoords, float3x4 worldToObject3x4, float3x4 objectToWorld3x4, int meshID, int primitiveID, float3 worldRayDirection, int instanceIndex)
{
    HitState hit;

    // Barycentric coordinate on the triangle
    const vec3 barycentrics = vec3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

    uint64_t primOffset = sizeof(PrimMeshInfo) * meshID;
    uint64_t vertAddress = vk::RawBufferLoad<uint64_t>(sceneDesc[0].primInfoAddress + primOffset);
    uint64_t indexAddress = vk::RawBufferLoad<uint64_t>(sceneDesc[0].primInfoAddress + primOffset + sizeof(uint64_t));

    uint64_t indexOffset = sizeof(uint3) * primitiveID;
    uint3 triangleIndex = vk::RawBufferLoad<uint3>(indexAddress + indexOffset);

    // Vertex and indices of the primitive
    Vertex v0 = getVertex(vertAddress, sizeof(Vertex) * triangleIndex.x);
    Vertex v1 = getVertex(vertAddress, sizeof(Vertex) * triangleIndex.y);
    Vertex v2 = getVertex(vertAddress, sizeof(Vertex) * triangleIndex.z);

    // Position
    const float3 pos0 = v0.position.xyz;
    const float3 pos1 = v1.position.xyz;
    const float3 pos2 = v2.position.xyz;
    const float3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
    const float4x4 matrix = vk::RawBufferLoad<int>(sceneDesc[0].instInfoAddress + sizeof(InstanceInfo) * instanceIndex);
    // hit.pos = mul(inverse(matrix), float4(position, 1.0f)).xyz; //
    hit.pos = float3(mul(objectToWorld3x4, float4(position, 1.0)));

    // Normal
    const float3 nrm0 = v0.normal.xyz;
    const float3 nrm1 = v1.normal.xyz;
    const float3 nrm2 = v2.normal.xyz;
    const float3 normal = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
    float3 worldNormal = normalize(mul(normal, worldToObject3x4).xyz);
    const float3 geoNormal = normalize(cross(pos1 - pos0, pos2 - pos0));
    float3 worldGeoNormal = normalize(mul(geoNormal, worldToObject3x4).xyz);
    hit.geonrm = worldGeoNormal;
    hit.nrm = worldNormal;

    // For low tessalated, avoid internal reflection
    vec3 r = reflect(normalize(worldRayDirection), hit.nrm);
    if (dot(r, hit.geonrm) < 0)
        hit.nrm = hit.geonrm;

    return hit;
}

//-----------------------------------------------------------------------
// Shoot a ray an return the information of the closest hit, in the
// PtPayload structure (PRD)
//
void traceRay(RayDesc ray, inout HitPayload payload)
{
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit(); // forcing to be opaque
    }

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {

        float2 barycentricCoords = q.CommittedTriangleBarycentrics();
        int meshID = q.CommittedInstanceID();          // rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
        int primitiveID = q.CommittedPrimitiveIndex(); // rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
        float3x4 worldToObject = q.CommittedWorldToObject3x4();
        float3x4 objectToWorld = q.CommittedObjectToWorld3x4();
        float hitT = q.CommittedRayT();
        int instanceIndex = q.CommittedInstanceIndex(); // rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);

        HitState hit = getHitState(barycentricCoords, worldToObject, objectToWorld, meshID, primitiveID, ray.Direction, instanceIndex);

        payload.hitT = hitT;
        payload.pos = hit.pos;
        payload.nrm = hit.nrm;
        payload.geonrm = hit.geonrm;
        payload.instanceIndex = instanceIndex;
    }
    else
    {
        payload.hitT = INFINITE;
    }
}

//-----------------------------------------------------------------------
// Shadow ray - return true if a ray hits anything
//
bool traceShadow(RayDesc ray)
{
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();
    return (q.CommittedStatus() != COMMITTED_NOTHING);
}

float3 getRandomPosition(float3 position, float radius, float2 randomValues)
{
    float angle = randomValues.x * 2.0 * 3.14159;
    float distance = sqrt(randomValues.y) * radius;

    float2 offset = float2(cos(angle), sin(angle)) * distance;
    float3 newPosition = float3(offset.x, 0, offset.y);

    return position + newPosition;
}

void setReconnectionSurface()
{
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
float3 pathTrace(RayDesc ray, inout uint seed, uint initSeed, float3 throughput, uint pixel1d, inout Reservoir res)
{
    float3 radiance = float3(0.0F, 0.0F, 0.0F);

    HitPayload payload;
    // InitialReservoir initRes = make();

    Sample s;
    {
        s.k = 0;
        s.primId = -1;
        s.p_hat_xi = float3(0.0f, 0.0f, 0.0f);
        s.seed = initSeed;
    }
    uint seed2 = seed;
    bool isSampleReady = false;
    float pathAliveP = 1.0f;
    // uint initialSeed = seed;
    float3 sampleThp = float3(0.0f, 0.0f, 0.0f); // 最初はThp無効(0に何かけても0なので)
    float3 LoCache = float3(0.0f, 0.0f, 0.0f);
    float3 thp_v0_to_vc = float3(0.0f, 0.0f, 0.0f); // 検証用 カメラからリコネクション点までのスループットをキャッシュする

    uint M = res.M;
    // Primary rayは発射済み
    for (int depth = 1; depth < pushConst.maxDepth; depth++)
    {
        int matID;
        // if (useGBuffer && depth == 0)
        // {
        //     GBufStruct gbuf = gbuffer1d[pixel1d];

        //     payload.nrm = gbuf.nrm;
        //     payload.pos = gbuf.pos;
        //     payload.hitT = gbuf.hitT;
        //     payload.geonrm = gbuf.nrm;
        //     matID = gbuf.matId;
        // }
        // else
        {
            // Retrieve the Instance buffer information
            traceRay(ray, payload);
            uint64_t materialIDOffest = sizeof(float4x4);
            uint64_t instOffset = sizeof(InstanceInfo) * payload.instanceIndex;
            matID = vk::RawBufferLoad<int>(sceneDesc[0].instInfoAddress + instOffset + materialIDOffest);
        }

        // Hitting the environment, then exit
        if (payload.hitT == INFINITE)
        {

            float3 sky_color = float3(0.0, 0.0, 0.0); // Light blue grey
            break;
            // return radiance + (sky_color * throughput);
        }

        float3 lightPos = getRandomPosition(pushConst.light.position, pushConst.light.radius, float2(rand(seed), rand(seed)));
        float distanceToLight = length(lightPos - payload.pos);

        float pdf = 0.0F;
        float3 V = -ray.Direction;
        float3 L = normalize(lightPos - payload.pos);

        // Retrieve the material color
        uint64_t matOffset = sizeof(Material) * matID;
        Material mat = getMaterial(sceneDesc[0].materialAddress, matOffset);

        // Setting up the material
        PbrMaterial pbrMat;
        pbrMat.albedo = float4(mat.albedo, 1);
        pbrMat.roughness = mat.roughness;
        pbrMat.metallic = mat.metallic;
        pbrMat.normal = payload.nrm;
        pbrMat.emissive = float3(0.0F, 0.0F, 0.0F);
        pbrMat.f0 = lerp(float3(0.04F, 0.04F, 0.04F), pbrMat.albedo.xyz, mat.metallic);

        { // Set GI Sample
            if (pbrMat.roughness > 0.1 && !isSampleReady)
            {
                // radianceは最後までパスとレースしないとわからない
                s.to.pos = payload.pos;
                s.to.nrm = payload.nrm;
                s.primId = payload.instanceIndex;
                s.k = depth;
                s.seed = initSeed;
                isSampleReady = true;
                // thpを初期化(有効化)
                sampleThp = float3(1.0f, 1.0f, 1.0f);
                thp_v0_to_vc = throughput;
            }
        }

        float3 contrib = float3(0, 0, 0);
        float3 Lo = float3(0.0f, 0.0f, 0.0f);

        // Evaluation of direct light (sun)
        bool nextEventValid = (dot(L, payload.nrm) > 0.0f);
        if (nextEventValid)
        {
            BsdfEvaluateData evalData;
            evalData.k1 = -ray.Direction;
            evalData.k2 = L;
            bsdfEvaluate(evalData, pbrMat);
            //  +
            const float3 w = (sceneDesc[0].light.intensity.xxx + float3(1.0f, 1.0f, 1.0f) * 100.0) / (distanceToLight * distanceToLight);
            contrib += w * evalData.bsdf_diffuse;
            contrib += w * evalData.bsdf_glossy;
            Lo = contrib;
            contrib *= throughput;
        }

        // Sample BSDF
        {
            BsdfSampleData sampleData;
            sampleData.k1 = -ray.Direction;                                         // outgoing direction
            sampleData.xi = float4(rand(seed), rand(seed), rand(seed), rand(seed)); // 4つめは今の実装だといらん

            bsdfSample(sampleData, pbrMat);
            if (sampleData.event_type == BSDF_EVENT_ABSORB)
            {
                break; // Need to add the contribution ?
            }

            throughput *= sampleData.bsdf_over_pdf;
            sampleThp *= sampleData.bsdf_over_pdf;

            ray.Origin = offsetRay(payload.pos, payload.geonrm);
            ray.Direction = sampleData.k2;
        }

        // Russian-Roulette (minimizing live state)
        float rrPcont = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001F, 0.95F);
        if (rand(seed) >= rrPcont)
            break;             // paths with low throughput that won't contribute
        throughput /= rrPcont; // boost the energy of the non-terminated paths
        sampleThp /= rrPcont;
        pathAliveP *= rrPcont;
        // We are adding the contribution to the radiance only if the ray is not occluded by an object.
        if (nextEventValid /* && depth != 0*/)
        {
            RayDesc shadowRay;
            shadowRay.Origin = ray.Origin;
            shadowRay.Direction = L;
            shadowRay.TMin = 0.01;
            shadowRay.TMax = distanceToLight;
            bool inShadow = traceShadow(shadowRay);
            if (!inShadow)
            {
                // InitialSample s;
                // s.radiance = contrib;
                // s.f = contrib;
                // updateReservoir(initRes, s, pathAliveP, rand(seed));
                if (isSampleReady)
                {
                    s.p_hat_xi = contrib;
                    s.p_hat_cached = Lo * sampleThp;
                    float w = toScalar(s.p_hat_xi) / pathAliveP;
                    updateReservoir(res, s, w, rand(seed2));
                }
                radiance += contrib;
            }
        }
        res.M = M;
    }
    res.M++;
    // float3 thp2 = isSampleReady ? sampleThp * thp_v0_to_vc : throughput;
    // return float3(isSampleReady, isSampleReady, isSampleReady);

    // return throughput - thp2;
    return radiance;
    // return res.s.radiance;
    // return throughput - thp_v0_to_vc;

    return res.s.p_hat_xi * calcContributionWegiht(res);
}

float3 evaluatePrimaryHit(uint2 pixel, uint pixel1d, inout RayDesc ray, inout uint seed, out uint initSeed, inout float3 throughput)
{
    initSeed = seed;
    float3 radiance = float3(0.0F, 0.0F, 0.0F);
    // float3 throughput = float3(1.0F, 1.0F, 1.0F);

    HitPayload payload;
    InitialReservoir initRes = make();
    Reservoir res = initReservoir();
    float weight = 1.0f;

    int matID;
    // if (useGBuffer && depth == 0)
    // {
    GBufStruct gbuf = gbuffer1d[pixel1d];
    payload.nrm = gbuf.nrm;
    payload.pos = gbuf.pos;
    payload.hitT = gbuf.hitT;
    payload.geonrm = gbuf.nrm;
    matID = gbuf.matId;
    // }
    // else
    // {
    //     // Retrieve the Instance buffer information
    //     traceRay(ray, payload);
    //     uint64_t materialIDOffest = sizeof(float4x4);
    //     uint64_t instOffset = sizeof(InstanceInfo) * payload.instanceIndex;
    //     matID = vk::RawBufferLoad<int>(sceneDesc[0].instInfoAddress + instOffset + materialIDOffest);
    // }

    // Hitting the environment, then exit
    if (payload.hitT == INFINITE)
    {

        float3 sky_color = float3(0.1, 0.1, 0.20); // Light blue grey
        return radiance + (sky_color * throughput);
    }

    float3 lightPos = getRandomPosition(pushConst.light.position, pushConst.light.radius, float2(rand(seed), rand(seed)));
    float distanceToLight = length(lightPos - payload.pos);

    float pdf = 0.0F;
    float3 V = -ray.Direction;
    float3 L = normalize(lightPos - payload.pos);

    // Retrieve the material color
    uint64_t matOffset = sizeof(Material) * matID;
    Material mat = getMaterial(sceneDesc[0].materialAddress, matOffset);

    // Setting up the material
    PbrMaterial pbrMat;
    pbrMat.albedo = float4(mat.albedo, 1);
    pbrMat.roughness = mat.roughness;
    pbrMat.metallic = mat.metallic;
    pbrMat.normal = payload.nrm;
    pbrMat.emissive = float3(0.0F, 0.0F, 0.0F);
    pbrMat.f0 = lerp(float3(0.04F, 0.04F, 0.04F), pbrMat.albedo.xyz, mat.metallic);

    float3 contrib = float3(0, 0, 0);
    float3 in_F = float3(0.0f, 0.0f, 0.0f);

    // Evaluation of direct light (sun)
    bool nextEventValid = (dot(L, payload.nrm) > 0.0f);
    if (nextEventValid)
    {
        BsdfEvaluateData evalData;
        evalData.k1 = -ray.Direction;
        evalData.k2 = L;
        bsdfEvaluate(evalData, pbrMat);

        const float3 w = (sceneDesc[0].light.intensity.xxx + float3(1.0f, 1.0f, 1.0f) * 500.0) / (distanceToLight * distanceToLight);
        contrib += w * evalData.bsdf_diffuse;
        contrib += w * evalData.bsdf_glossy;
        in_F = contrib;
        contrib *= throughput;
    }

    // Sample BSDF
    {
        BsdfSampleData sampleData;
        sampleData.k1 = -ray.Direction;                                         // outgoing direction
        sampleData.xi = float4(rand(seed), rand(seed), rand(seed), rand(seed)); // 4つめは今の実装だといらん

        bsdfSample(sampleData, pbrMat);
        if (sampleData.event_type == BSDF_EVENT_ABSORB)
        {
            return radiance;
            // break; // Need to add the contribution ?
        }

        throughput *= sampleData.bsdf_over_pdf;

        ray.Origin = offsetRay(payload.pos, payload.geonrm);
        ray.Direction = sampleData.k2;
    }

    // Russian-Roulette (minimizing live state)
    float rrPcont = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001F, 0.95F);
    if (rand(seed) >= rrPcont)
        return radiance;
    // break;             // paths with low throughput that won't contribute
    throughput /= rrPcont; // boost the energy of the non-terminated paths
    weight *= rrPcont;
    // We are adding the contribution to the radiance only if the ray is not occluded by an object.
    if (nextEventValid)
    {
        RayDesc shadowRay;
        shadowRay.Origin = ray.Origin;
        shadowRay.Direction = L;
        shadowRay.TMin = 0.01;
        shadowRay.TMax = distanceToLight;
        bool inShadow = traceShadow(shadowRay);
        if (!inShadow)
        {
            InitialSample s;
            s.radiance = contrib;
            s.f = contrib;
            updateReservoir(initRes, s, weight, rand(seed));
            radiance += contrib;
        }
    }
    return radiance;
}

//-----------------------------------------------------------------------
// Sampling the pixel
//-----------------------------------------------------------------------
float3 samplePixel(inout uint seed, float2 launchID, float2 launchSize, inout Reservoir res)
{
    // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
    const float2 subpixel_jitter = pushConst.frame == 0 ? float2(0.5f, 0.5f) : float2(rand(seed), rand(seed));
    const float2 pixelCenter = launchID + subpixel_jitter;
    const float2 inUV = pixelCenter / launchSize;
    const float2 d = inUV * 2.0 - 1.0;
    const float4 target = mul(frameInfo.projInv, float4(d.x, d.y, 0.01, 1.0));
    float3 rayDir = mul(frameInfo.viewInv, float4(normalize(target.xyz), 0.0)).xyz;

    uint pixel1d = launchID.x + launchSize.x * launchID.y;
    // float3 thp = evaluatePrimaryHit(launchID, pixel1d, rayDir, seed);

    RayDesc ray;
    ray.Origin = mul(frameInfo.viewInv, float4(0.0, 0.0, 0.0, 1.0)).xyz;
    ray.Direction = rayDir;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;
    float3 thp = float3(1.0f, 1.0f, 1.0f);
    uint initSeed;
    float3 di_radiance = evaluatePrimaryHit(launchID, pixel1d, ray, seed, initSeed, thp);
    float3 gi_radiance = pathTrace(ray, seed, initSeed, thp, pixel1d, res);
    float3 radiance = gi_radiance;

    // Removing fireflies
    float lum = dot(radiance, float3(0.212671F, 0.715160F, 0.072169F));
    if (lum > pushConst.fireflyClampThreshold)
    {
        radiance *= pushConst.fireflyClampThreshold / lum;
    }

    return radiance;
}

//-----------------------------------------------------------------------
// RAY GENERATION
//-----------------------------------------------------------------------
[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void computeMain(uint3 threadIdx: SV_DispatchThreadID)
{
    uint2 launchID = threadIdx.xy;

    uint2 imgSize;
    outImage.GetDimensions(imgSize.x, imgSize.y); // DispatchRaysDimensions();
    // float2 launchSize = imgSize;
    uint pixel1d = launchID.x + imgSize.x * launchID.y;
    if (launchID.x >= imgSize.x || launchID.y >= imgSize.y)
        return;

    // Initialize the random number
    uint seed = xxhash32(uint3(launchID.xy, pushConst.frame));

    // Sampling n times the pixel
    float3 pixel_color = float3(0.0F, 0.0F, 0.0F);
    Reservoir res = initReservoir();
    for (int s = 0; s < pushConst.maxSamples; s++)
    {
        pixel_color += samplePixel(seed, launchID, (float2)imgSize, res);
    }
    packedReservoir[pixel1d] = pack(res);

    // pixel_color /= pushConst.maxSamples;
    bool first_frame = (pushConst.frame == 0);
    // Saving result
    if (true)
    {                                                // First frame, replace the value in the buffer
        diReservoir[pixel1d].radiance = pixel_color; // gbuffer1d[pixel1d].nrm; // pixel_color;
        // diReservoir[pixel1d].radiance = gbuffer1d[pixel1d].matId; // pixel_color;
    }
    else
    { // Do accumulation over time
        float a = 1.0F / float(pushConst.frame + 1);
        float3 old_color = diReservoir[pixel1d].radiance;
        diReservoir[pixel1d].radiance = lerp(old_color, pixel_color, a);
    }
}

