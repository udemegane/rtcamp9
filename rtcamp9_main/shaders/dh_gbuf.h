#pragma once
#include "constants.hlsli"

#ifndef GBUF_BINDING
#define GBUF_BINDING 1

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

#define B_gbuf_tlas 0
#define B_gbuf_outBuffer 1
#define B_gbuf_frameInfo 2
#define B_gbuf_sceneDesc 3

struct GbufConst
{
    uint width;
    uint height;
};

struct GBufStruct
{
    vec3 pos;
    vec3 nrm;
    int matId;
    float hitT;
};

#endif