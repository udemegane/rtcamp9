#include "device_host.h"
#include "dh_gbuf.h"

// [[vk::push_constant]]
// ConstantBuffer<GbufConst> constant;
[[vk::binding(B_gbuf_tlas)]]
RaytracingAccelerationStructure topLevelAS;
[[vk::binding(B_gbuf_outBuffer)]]
ConstantBuffer<GBufStruct> gbuf;
[[vk::binding(B_gbuf_frameInfo)]]
ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(B_gbuf_sceneDesc)]]
StructuredBuffer<SceneDescription> sceneDesc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID, uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    uint2 imgSize;
}
