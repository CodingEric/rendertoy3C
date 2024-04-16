#pragma once

#include "wavefront.h"

namespace rendertoy3o {

enum class MaterialType
{
    Diffuse,
    Specular,
    FresnelTransmissive,
    Principled
};

struct Material
{
    MaterialType m_materialType;

    float3 m_diffuse{1.0f, 1.0f, 1.0f};
    int m_diffuseTextureID{-1};

    float3 m_emissive{0.0f, 0.0f, 0.0f};
    int m_emissiveTextureID{-1};

    float m_roughness{0.5f};
    int m_roughnessTextureID{-1};

    float m_anisotropy{0.0f};
    float m_ior{1.333f};
    float m_transmittance{0.0f};

    int m_normalTextureID{-1};

    // WAVEFRONT_GPU BSDF GetBSDF()
    // {

    // }
};

} // namespace rendertoy3o