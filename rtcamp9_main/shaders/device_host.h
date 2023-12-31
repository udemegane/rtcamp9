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

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

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
#define GROUP_SIZE 16
#define SUB_FRAMES 100

#define USE_SPATIAL false
#define USE_TEMPORAL true
struct Light
{
  vec3 position;
  vec3 color;
  float intensity;
  float radius; // on XZ plane
};

struct PushConstant
{
  int maxDepth;
  int frame;
  float fireflyClampThreshold;
  int maxSamples;
  Light light;
  int subFrame;
  int maxSubframes;
  int _1;
  int _2;
};

struct FrameInfo
{
  mat4 proj;
  mat4 view;
  mat4 projInv;
  mat4 viewInv;
  mat4 prevProj;
  mat4 prevView;
  mat4 prevProjInv;
  mat4 prevViewInv;
  vec3 camPos;
  int width;
  int height;
  int frame;
};

struct Material
{
  vec3 albedo;
  float roughness;
  float metallic;
  float ior;
};

// From primitive
struct Vertex
{
  vec3 position;
  vec3 normal;
  vec2 t;
};

struct PrimMeshInfo
{
  uint64_t vertexAddress;
  uint64_t indexAddress;
};

struct InstanceInfo
{
  mat4 transform;
  int materialID;
};

struct SceneDescription
{
  uint64_t materialAddress;
  uint64_t instInfoAddress;
  uint64_t primInfoAddress;
  Light light;
};

#endif // HOST_DEVICE_H
