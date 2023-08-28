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

// #define GROUP_SIZE 16

struct DIReservoir
{
    vec3 radiance;
    float _dummy;
};

struct PackedReservoir
{
    // 32bit x 16 = 64B
    vec3 pos;
    vec3 nrm;
    vec3 rad;
    int primId;
    uint k;
    uint seed;
    float w;
    float wSum;
    uint M;
    uint _dummy;
};