#ifndef DH_VIS_H
#define DH_VIS_H 1

#ifdef __cplusplus
using uint = uint32_t;
using vec3 = nvmath::vec3f;
using vec2 = nvmath::vec2f;
#define INLINE inline
#else
#define INLINE
#endif

#define GROUP_SIZE 16

// Bindings
#define eDebugPassInput 0
#define eDebugPassOutput 1

struct DBGConstant
{
    float dummy;
    // int maxDepth;
    // int frame;
    // float fireflyClampThreshold;
    // int maxSamples;
    // Light light;
};

#endif