
#pragma once


#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <set>
#include <tuple>

#include <optix.h>
#include <sutil/vec_math.h>
#include <cuda/helpers.h>

struct Mesh
{
    std::vector<float3> vertices;
    std::vector<int3> indices;
    std::vector<float3> normals;
    std::vector<float2> texcoords;

    float3 diffuse { 1.0f, 1.0f, 1.0f };
    int diffuseTextureID { -1 };

    float3 emissive { 0.0f, 0.0f, 0.0f };
    int emissiveTextureID { -1 };

    float roughness { 0.5f };
    int roughnessTextureID { -1 };

    float anisotropy { 0.0f };

    float ior { 1.333f };

    float transmittance { 0.0f };

    int normalTextureID { -1 };
};

struct Texture
{
    std::vector<uint32_t> pixel = {};
    int2 resolution { -1, -1 };
};

std::tuple<std::vector<Mesh>, std::vector<Texture> > loadOBJ(const std::string &path);
