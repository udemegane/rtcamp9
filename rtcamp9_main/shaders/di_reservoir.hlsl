#ifdef __cplusplus
using mat4 = nvmath::mat4f;
using vec4 = nvmath::vec4f;
using vec3 = nvmath::vec3f;
using vec2 = nvmath::vec2f;
using uint = uint32_t;
#else
#define mat4 float4x4
#define vec4 float4
#define vec3 float3
#define vec2 float2
#endif // __cplusplus

struct DIReservoir
{
    vec3 radiance;
};
