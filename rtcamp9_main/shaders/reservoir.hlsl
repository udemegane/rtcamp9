#pragma once
#include "constants.hlsli"
#include "dh_reservoir.hlsl"

static const uint Mcap = 20;

float toScalar(float3 color)
{
    return dot(color, float3(0.299, 0.587, 0.114)); // luminance
}
struct RcVertex
{
    float3 pos;
    float3 nrm;
};
// Sample for Path-Resampling
struct Sample
{
    // RcVertex from; // Reconnection-able origin Vertex V_{k-1}
    RcVertex to;         // Reconnection-able destination Vertex V_k
    int primId;          //
    float3 p_hat_xi;     // radiance of contribution from V_emit to V_0.
    float3 p_hat_cached; // cached radiance of contribution from V_emit to V_k
    uint k;              // Index of Reconnection-able Vertex
    uint seed;           // seed of puesdo random generator
    // float3 u1;       // 3 random [0,1] for BSDF Random-Replay Sampling of Voffset_{k-2}, which was generated and used Vbase_{k-2}. (Ideally should we cache all randoms from Vbase_1 to Vbase_{k-2}?)
    // float3 u2;       // 3 random [0,1] for BSDF Sampling of V_k. If scene objects is dynamic, you need to launch shadow ray from V_k with this.
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

Reservoir initReservoir()
{
    Reservoir r;
    Sample s;
    s.k = 0;
    s.seed = 0;
    s.p_hat_xi = float3(0.0f, 0.0f, 0.0f);
    s.primId = -1;
    s.to.nrm = float3(1.0f, 0.0f, 0.0f);
    s.to.pos = float3(0.0f, 0.0f, 0.0f);
    r.s = s;
    r.w = 0.0f;
    r.wSum = 0.0f;
    r.M = 0;
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
    r.M++;
    if (w_i == 0)
        return false;
    // float w_i = toScalar(s_i.radiance) / p_i;
    // float w_current = toScalar(r.s.radiance) / r.p;
    r.wSum += w_i;

    // Accept?
    if (r.w * u <= w_i)
    {
        r.s = s_i;
        r.w = w_i;
        return true;
    }
    return false;
}

bool updateReservoir(inout Reservoir r, in Sample s_i, const float w_i, uint M, const float u)
{
    r.M += M;
    if (w_i == 0)
        return false;
    // float w_i = toScalar(s_i.radiance) / p_i;
    // float w_current = toScalar(r.s.radiance) / r.p;
    r.wSum += w_i;

    // Accept?
    if (r.w * u <= w_i)
    {
        r.s = s_i;
        r.w = w_i;
        return true;
    }
    return false;
}

// float updateReservoir(inout Reservoir r, in Sample s_i, const float p_i, float w, const float u)
// {
//     r.M++;
//     if (p_i == 0)
//         return 0.0f;
//     float w_i = toScalar(s_i.radiance) / p_i;
//     if (w_i != w)
//         return -1;
//     float correct_p_inv = r.p / toScalar(r.s.radiance);
//     float w_current = toScalar(r.s.radiance) * correct_p_inv; // おかしい
//     float tmp = toScalar(r.s.radiance) * r.p;
//     r.wSum += w_i;

//     // Accept?
//     if (tmp * u <= w_i || r.M == 0)
//     {
//         r.s = s_i;
//         r.p = w;
//         return 1.0f;
//     }
//     return 0.0f;
// }

void capReservoir(inout Reservoir r)
{

    r.M = min(r.M, Mcap);
}

float calcContributionWegiht(Reservoir r)
{
    float p_hat = length(r.s.p_hat_xi);
    return p_hat == 0.0f ? 0.0f : (r.wSum / (r.M * (p_hat + FLT_EPSILON)));
}

// bool updateReservoir(inout DIReservoir r, in Sample)

float calcJacobian(RcVertex from, RcVertex to)
{

    return 0.f;
}

// struct PackedReservoir
// {
//     float4 posnrmx1;
//     float4 posnrmx2;
//     float4 nrmyz;
//     float4 u1w;
//     // float4 u2wsum;
//     float4 radWsum;
//     int primId;
//     uint k;
//     uint M;
//     uint _dummy;
//     // float r1;
//     // float r2;
//     // float r3;
// };

PackedReservoir pack(Reservoir r)
{
    PackedReservoir pr;
    pr.pos = r.s.to.pos;
    pr.nrm = r.s.to.nrm;
    pr.rad = r.s.p_hat_xi;
    pr.radTmp = r.s.p_hat_cached;
    pr.primId = r.s.primId;
    pr.k = r.s.k;
    pr.seed = r.s.seed;
    pr.w = r.w;
    pr.wSum = r.wSum;
    pr.M = r.M;
    pr._dummy = 0u;
    return pr;
};

Reservoir unpack(PackedReservoir pr)
{
    Reservoir r;
    r.s.to.pos = pr.pos;
    r.s.to.nrm = pr.nrm;
    r.s.p_hat_xi = pr.rad;
    r.s.p_hat_cached = pr.radTmp;
    r.s.primId = pr.primId;
    r.s.k = pr.k;
    r.s.seed = pr.seed;
    r.w = pr.w;
    r.wSum = pr.wSum;
    r.M = pr.M;
    return r;
};
