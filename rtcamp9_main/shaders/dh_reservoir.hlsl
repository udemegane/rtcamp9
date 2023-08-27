#pragma once
#include "constants.hlsli"

#ifdef __cplusplus
using mat4 = nvmath::mat4f;
using vec4 = nvmath::vec4f;
using vec3 = nvmath::vec3f;
using vec2 = nvmath::vec2f;
using uint = uint32_t;
#else
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
#endif // __cplusplus

struct DIReservoir
{
    vec3 radiance;
};

struct PackedReservoir
{
    vec4 posnrmx1;
    vec4 posnrmx2;
    vec4 nrmyz;
    vec4 u1w;
    // vec4 u2wsum;
    vec4 radWsum;
    int primId;
    uint k;
    uint M;
    uint _dummy;
    // float r1;
    // float r2;
    // float r3;
};