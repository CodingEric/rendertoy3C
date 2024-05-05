#define TINYOBJLOADER_IMPLEMENTATION

#include <iostream>
#include <algorithm>
#include <set>
#include <ranges>

#include <support/tinyobj/tiny_obj_loader.h>
#define STB_IMAGE_IMPLEMENTATION
#include <support/stb/stb_image.h>
#include "mesh.h"

struct Compare
{
    inline bool operator()(const tinyobj::index_t &a,
                           const tinyobj::index_t &b) const
    {
        if (a.vertex_index < b.vertex_index)
            return true;
        if (a.vertex_index > b.vertex_index)
            return false;

        if (a.normal_index < b.normal_index)
            return true;
        if (a.normal_index > b.normal_index)
            return false;

        if (a.texcoord_index < b.texcoord_index)
            return true;
        if (a.texcoord_index > b.texcoord_index)
            return false;

        return false;
    }
};

[[nodiscard]] std::tuple<std::vector<rendertoy3o::Mesh>, std::vector<rendertoy3o::Texture>> rendertoy3o::loadOBJ(const std::vector<std::string> &paths)
{
    const auto key_frames = paths.size();
    std::vector<tinyobj::ObjReader> readers;
    readers.resize(key_frames);
    for (unsigned int i = 0; i < key_frames; ++i)
    {
        auto &reader = readers[i];
        auto &path = paths[i];
        if (!reader.ParseFromFile(path)) [[unlikely]]
        {
            if (!reader.Error().empty()) [[likely]]
                std::cerr << "TinyObjReader: " << reader.Error();
            exit(1);
        }

        if (!reader.Warning().empty()) [[unlikely]]
            std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto &shapes = readers[0].GetShapes();
    auto &materials = readers[0].GetMaterials();

    std::vector<rendertoy3o::Mesh> ret_mesh = {};
    std::vector<rendertoy3o::Texture> ret_texture = {};

    for (const auto &shape : shapes)
    {
        std::set<int> materialIDs;
        for (auto faceMatID : shape.mesh.material_ids)
        {
            materialIDs.insert(faceMatID);
        }

        for (int materialID : materialIDs)
        {
            std::map<tinyobj::index_t, int, Compare> knownVertices;
            std::map<std::string, int> knownTextures;
            rendertoy3o::Mesh mesh = {};
            mesh.vertices.resize(key_frames);
            mesh.normals.resize(key_frames);
            mesh.texcoords.resize(key_frames);

            auto addVertexAndGetIndexInMesh = [&](tinyobj::index_t vertexIndexInOBJ) -> int
            {
                if (knownVertices.find(vertexIndexInOBJ) != knownVertices.end())
                {
                    return knownVertices[vertexIndexInOBJ];
                }

                int indexInMesh = mesh.vertices[0].size();
                knownVertices[vertexIndexInOBJ] = indexInMesh;

                for (unsigned int i = 0; i < key_frames; ++i)
                {
                    auto &attrib = readers[i].GetAttrib();
                    const float3 *vertex_array = (const float3 *)attrib.vertices.data();
                    const float3 *normal_array = (const float3 *)attrib.normals.data();
                    const float2 *texcoord_array = (const float2 *)attrib.texcoords.data();
                    mesh.vertices[i].push_back(vertex_array[vertexIndexInOBJ.vertex_index]);
                    if (vertexIndexInOBJ.normal_index >= 0)
                    {
                        while (mesh.normals[i].size() < mesh.vertices[i].size())
                            mesh.normals[i].push_back(normal_array[vertexIndexInOBJ.normal_index]);
                    }
                    if (vertexIndexInOBJ.texcoord_index >= 0)
                    {
                        while (mesh.texcoords[i].size() < mesh.vertices[i].size())
                            mesh.texcoords[i].push_back(texcoord_array[vertexIndexInOBJ.texcoord_index]);
                    }
                }

                return indexInMesh;
            };

            auto addTextureAndGetTextureId = [&](const std::string inFilename, const std::string &modelpath) -> int
            {
                if (inFilename == "")
                {
                    return -1;
                }

                if (knownTextures.find(inFilename) != knownTextures.end())
                {
                    return knownTextures[inFilename];
                }

                auto filename = inFilename;
                for (auto &c : filename)
                {
                    if (c == '\\')
                    {
                        c = '/';
                    }
                }

                filename = modelpath + "/" + filename;

                int2 res;
                int comp;
                unsigned char *image = stbi_load(filename.c_str(), &res.x, &res.y, &comp, STBI_rgb_alpha);
                int textureID = -1;
                if (image)
                {
                    textureID = ret_texture.size();
                    rendertoy3o::Texture tex;
                    tex.resolution = res;
                    tex.pixel.resize(res.x * res.y);
                    auto imagePixelView = (uint32_t *)image;
                    for (int i = 0; i < res.x * res.y; ++i)
                    {
                        tex.pixel[i] = imagePixelView[i];
                    }

                    for (int y = 0; y < res.y / 2; ++y)
                    {
                        uint32_t *line_y = tex.pixel.data() + y * res.x;
                        uint32_t *mirrored_y = tex.pixel.data() + (res.y - 1 - y) * res.x;
                        for (int x = 0; x < res.x; x++)
                        {
                            std::swap(line_y[x], mirrored_y[x]);
                        }
                    }

                    ret_texture.push_back(tex);
                }
                else
                {
                    std::cerr << "Error loading texture." << std::endl;
                }

                knownTextures[inFilename] = textureID;
                return textureID;
            };

            const std::string modelDir = paths[0].substr(0, paths[0].rfind('/') + 1);

            for (size_t faceID = 0; faceID < shape.mesh.material_ids.size(); ++faceID)
            {
                if (shape.mesh.material_ids[faceID] != materialID)
                    continue;
                tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                int3 idx{addVertexAndGetIndexInMesh(idx0), addVertexAndGetIndexInMesh(idx1), addVertexAndGetIndexInMesh(idx2)};
                mesh.indices.push_back(idx);

                // TODO: handle null material.
                mesh.material.m_diffuse = (const float3 &)materials[materialID].diffuse;
                mesh.material.m_diffuseTextureID = addTextureAndGetTextureId(materials[materialID].diffuse_texname, modelDir);

                mesh.material.m_emissive = (const float3 &)materials[materialID].emission;
                mesh.material.m_emissiveTextureID = addTextureAndGetTextureId(materials[materialID].emissive_texname, modelDir);

                mesh.material.m_roughness = (const float &)materials[materialID].roughness;
                mesh.material.m_roughnessTextureID = addTextureAndGetTextureId(materials[materialID].roughness_texname, modelDir);

                mesh.material.m_anisotropy = (const float &)materials[materialID].anisotropy;
                mesh.material.m_ior = (const float &)materials[materialID].ior;
                mesh.material.m_transmittance = (const float &)materials[materialID].transmittance;
                mesh.material.m_normalTextureID = addTextureAndGetTextureId(materials[materialID].normal_texname, modelDir);
            }

            if (!mesh.vertices[0].empty())
            {
                mesh.num_keys = key_frames;
                ret_mesh.push_back(mesh);
            }
        }
    }

    return {ret_mesh, ret_texture};
}
