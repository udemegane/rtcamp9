#pragma once
#include "constants.hlsli"

struct RcVertex
{
    float3 pos;
    float3 nrm;
};
// Sample for Path-Resampling
struct Sample
{
    RcVertex from; // Reconnection-able origin Vertex V_{k-1}
    RcVertex to;   // Reconnection-able destination Vertex V_k
    int primId;
    float3 radiance; // cached radiance of contribution from V_emit to V_k.
    uint k;          // Index of Reconnection-able Vertex
    float3 u1;       // 3 random [0,1] for BSDF Random-Replay Sampling of Voffset_{k-2}, which was generated and used Vbase_{k-2}. (Ideally should we cache all randoms from Vbase_1 to Vbase_{k-2}?)
    float3 u2;       // 3 random [0,1] for BSDF Sampling of V_k. If scene objects is dynamic, you need to launch shadow ray from V_k with this.
};

// Path-Resampling Reservoir
struct Reservoir
{
    Sample s;   // Selected Sample by RIS
    float w;    // Resampling weight of RIS
    float wSum; // Sum of resampling weight
    uint M;     // Size of reservoir
    // float c; // Contribution MIS weights
    // float m; //new resampling MIS weights
};

struct InitialSample
{
    float3 radiance;
    float3 f;
};
struct InitialReservoir
{
    InitialSample s;
    float w;
    float wSum;
};

InitialReservoir make()
{
    InitialReservoir r;
    InitialSample s;
    s.radiance = float3(0.f, 0.f, 0.f);
    s.f = float3(0.f, 0.f, 0.f);
    r.s = s;
    r.w = 0.0f;
    r.wSum = 0.0f;
    return r;
}

// Update Path-Reservoir with stream input s and weight.
// This is used when Initial Path-Resampling(NEE only).
bool updateReservoir(inout InitialReservoir r, in InitialSample s_i, const float w_i, const float u)
{
    // r.M++;
    if (w_i == 0)
        return false;
    r.w += w_i;

    // Accept?
    if (r.w * u <= w_i)
    {
        r.s = s_i;
        return true;
    }
    return false;
}

float calcContributionWegiht(InitialReservoir r)
{
    float p_hat = length(r.s.radiance);
    return p_hat == 0.0f ? 0.0f : (r.wSum / (p_hat + FLT_EPSILON));
}

bool updateReservoir(inout Reservoir r, in Sample s_i, const float w_i, const float u)
{
    return false;
}

// bool updateReservoir(inout DIReservoir r, in Sample)

bool mergeReservoirFromDifferentDomains(inout Reservoir base_r, Reservoir in_r, float u)
{
    return false;
}

float calcJacobian(RcVertex from, RcVertex to)
{
    return 0.f;
}

// TODO: implement correct this
struct PackedReservoir
{
    float4 posnrmx1;
    float4 posnrmx2;
    float4 nrmyz;
    float4 u1w;
    float4 u2wsum;
    // int primId;
    uint k;
    // uint M;
    float r1;
    float r2;
    float r3;
};

PackedReservoir pack(Reservoir r)
{
    PackedReservoir outR;
    outR.posnrmx1.xyz = r.s.from.pos;
    outR.posnrmx1.w = r.s.from.nrm.x;
    outR.posnrmx2.xyz = r.s.to.pos;
    outR.posnrmx2.w = r.s.to.nrm.x;
    outR.nrmyz.xy = r.s.from.nrm.yz;
    outR.nrmyz.zw = r.s.to.nrm.yz;
    outR.u1w.xyz = r.s.u1;
    outR.u1w.w = r.w;
    outR.u2wsum.xyz = r.s.u2;
    outR.u2wsum.w = r.wSum;
    outR.k = r.s.k;
    outR.r1 = r.s.radiance.x;
    outR.r2 = r.s.radiance.y;
    outR.r3 = r.s.radiance.z;
    return outR;
};

Reservoir unpack(PackedReservoir r)
{
    Reservoir outR;
    outR.s.from.pos = r.posnrmx1.xyz;
    outR.s.from.nrm.x = r.posnrmx1.w;
    outR.s.to.pos = r.posnrmx2.xyz;
    outR.s.to.nrm.x = r.posnrmx2.w;
    outR.s.from.nrm.yz = r.nrmyz.xy;
    outR.s.to.nrm.yz = r.nrmyz.zw;
    outR.s.primId = -1;
    outR.s.k = r.k;

    return outR;
};
