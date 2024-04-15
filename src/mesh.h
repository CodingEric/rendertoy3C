#pragma once

#include <vector>
#include <string>
#include <tuple>

#include "wavefront.h"
#include "material.h"

namespace wavefront {

struct Mesh
{
    unsigned int num_keys{1};

    std::vector<std::vector<float3>> vertices;
    std::vector<int3> indices;
    std::vector<std::vector<float3>> normals;
    std::vector<std::vector<float2>> texcoords;

    Material material;
};

struct Texture
{
    std::vector<uint32_t> pixel = {};
    int2 resolution { -1, -1 };
};

std::tuple<std::vector<Mesh>, std::vector<Texture> > loadOBJ(const std::vector<std::string> &path);

} // namespace wavefront
