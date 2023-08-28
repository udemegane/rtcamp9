#include "device_host.h"
#include "dh_gbuf.h"
#include "ggx.hlsli"
#include "random.hlsli"

[[vk::push_constant]]
ConstantBuffer<GbufConst> constant;
[[vk::binding(B_gbuf_tlas)]]
RaytracingAccelerationStructure topLevelAS;
[[vk::binding(B_gbuf_outBuffer)]]
RWStructuredBuffer<GBufStruct> gbuffer1d;
[[vk::binding(B_gbuf_frameInfo)]]
ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(B_gbuf_sceneDesc)]]
StructuredBuffer<SceneDescription> sceneDesc;

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

GBufStruct samplePixel(inout uint seed, float2 launchID, float2 launchSize)
{
    // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
    const float2 subpixel_jitter = float2(rand(seed), rand(seed));
    const float2 pixelCenter = launchID + subpixel_jitter;
    const float2 inUV = pixelCenter / launchSize;
    const float2 d = inUV * 2.0 - 1.0;
    const float4 target = mul(frameInfo.projInv, float4(d.x, d.y, 0.01, 1.0));

    RayDesc ray;
    ray.Origin = mul(frameInfo.viewInv, float4(0.0, 0.0, 0.0, 1.0)).xyz;
    float3 raydir = mul(frameInfo.viewInv, float4(normalize(target.xyz), 0.0)).xyz;
    ray.Direction = raydir;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    HitPayload payload;
    traceRay(ray, payload);
    GBufStruct gbuf;
    gbuf.hitT = payload.hitT;
    if (payload.hitT >= INFINITE)
    {
        // gbuf.nrm = raydir;
        return gbuf;
    }

    // Retrieve the Instance buffer information
    uint64_t materialIDOffest = sizeof(float4x4);
    uint64_t instOffset = sizeof(InstanceInfo) * payload.instanceIndex;
    int matID = vk::RawBufferLoad<int>(sceneDesc[0].instInfoAddress + instOffset + materialIDOffest);

    gbuf.pos = payload.pos;
    gbuf.nrm = payload.nrm;
    gbuf.matId = matID;
    // gbuf.nrm.x = launchID.x / launchSize.x;
    // gbuf.nrm.y = launchID.y / launchSize.y;
    // gbuf.nrm = float3(0.0f, 1.0f, 0.0f);

    return gbuf;
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
    gbuffer1d[pixel1d] = samplePixel(seed, pixel, imgSize);
}
