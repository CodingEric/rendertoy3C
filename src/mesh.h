#pragma once

#include <vector>
#include <string>
#include <tuple>

#include "wavefront.h"
#include "material.h"

struct Mesh
{
    std::vector<float3> vertices;
    std::vector<int3> indices;
    std::vector<float3> normals;
    std::vector<float2> texcoords;

    Material material;
};

struct Texture
{
    std::vector<uint32_t> pixel = {};
    int2 resolution { -1, -1 };
};

std::tuple<std::vector<Mesh>, std::vector<Texture> > loadOBJ(const std::string &path);
