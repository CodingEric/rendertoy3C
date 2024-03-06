#define TINYOBJLOADER_IMPLEMENTATION

#include <iostream>
#include <algorithm>
#include <set>

#include <support/tinyobj/tiny_obj_loader.h>
#include <support/stb/stb_image.h>
#include "mesh.h"

struct Compare
{
    inline bool operator()(const tinyobj::index_t &a,
                           const tinyobj::index_t &b)
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

std::tuple<std::vector<wavefront::Mesh>, std::vector<wavefront::Texture>> wavefront::loadOBJ(const std::string &path)
{
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path))
    {
        if (!reader.Error().empty())
        {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty())
    {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();
    auto &materials = reader.GetMaterials();

    std::vector<wavefront::Mesh> ret_mesh = {};
    std::vector<wavefront::Texture> ret_texture = {};

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
            wavefront::Mesh mesh = {};

            auto addVertexAndGetIndexInMesh = [&](tinyobj::index_t vertexIndexInOBJ) -> int
            {
                if (knownVertices.find(vertexIndexInOBJ) != knownVertices.end())
                {
                    return knownVertices[vertexIndexInOBJ];
                }

                const float3 *vertex_array = (const float3 *)attrib.vertices.data();
                const float3 *normal_array = (const float3 *)attrib.normals.data();
                const float2 *texcoord_array = (const float2 *)attrib.texcoords.data();

                int indexInMesh = mesh.vertices.size();
                knownVertices[vertexIndexInOBJ] = indexInMesh;

                mesh.vertices.push_back(vertex_array[vertexIndexInOBJ.vertex_index]);
                if (vertexIndexInOBJ.normal_index >= 0)
                {
                    while (mesh.normals.size() < mesh.vertices.size())
                        mesh.normals.push_back(normal_array[vertexIndexInOBJ.normal_index]);
                }
                if (vertexIndexInOBJ.texcoord_index >= 0)
                {
                    while (mesh.texcoords.size() < mesh.vertices.size())
                        mesh.texcoords.push_back(texcoord_array[vertexIndexInOBJ.texcoord_index]);
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
                    wavefront::Texture tex;
                    tex.resolution = res;
                    tex.pixel.resize(res.x * res.y);
                    auto imagePixelView = (uint32_t *)image;
                    for(int i = 0; i < res.x * res.y; ++i)
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

            const std::string modelDir = path.substr(0, path.rfind('/') + 1);

            for (size_t faceID = 0; faceID < shape.mesh.material_ids.size(); ++faceID)
            {
                if (shape.mesh.material_ids[faceID] != materialID)
                    continue;
                tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                int3 idx{addVertexAndGetIndexInMesh(idx0), addVertexAndGetIndexInMesh(idx1), addVertexAndGetIndexInMesh(idx2)};
                mesh.indices.push_back(idx);
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

            if (!mesh.vertices.empty())
            {
                ret_mesh.push_back(mesh);
            }
        }
    }

    return {ret_mesh, ret_texture};
}
