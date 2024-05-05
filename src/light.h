#pragma once

#include "wavefront.h"
#include "random.h"

namespace rendertoy3o {

enum class LightType
{
    SurfaceLight,
};

struct Light
{
    LightType m_lightType;

    float3 m_emission;

    float3 m_v0, m_v1, m_v2;
    float3 m_normal;
    float m_area;

    RENDERTOY_CPU_GPU
    Light(float3 emission, float3 v0, float3 v1, float3 v2, LightType lightType = LightType::SurfaceLight) noexcept
        : m_lightType(lightType), m_emission(emission), m_v0(v0), m_v1(v1), m_v2(v2)
    {
        m_normal = cross(m_v1 - m_v0, m_v2 - m_v0);
        m_area = 0.5f * length(m_normal);
        m_normal = normalize(m_normal);
    }

    RENDERTOY_GPU
    void Sample(const float3 &P, unsigned int &seed, float3 &light_pos, float3 &emission, float &pdf) noexcept
    {
        const float u = rnd(seed);
        const float v = rnd(seed);
        float su0 = std::sqrt(u);
        float b0 = 1.0f - su0;
        float b1 = v * su0;
        light_pos = b0 * m_v0 + b1 * m_v1 + (1.0f - b0 - b1) * m_v2;

        float3 dist_vec = light_pos - P;
        float dist2 = dot(dist_vec, dist_vec);
        if (dist2 < 1e-5f)
        {
            emission = {0.0f, 0.0f, 0.0f};
            pdf = 1.0f;
            return;
        }
        float3 normalized_dist_vec = normalize(dist_vec);
        float omega_light = abs(dot(normalized_dist_vec, m_normal)) * m_area / dist2;
        if (omega_light < 1e-5f)
        {
            emission = {0.0f, 0.0f, 0.0f};
            pdf = 1.0f;
            return;
        }
        emission = m_emission * omega_light;
        pdf = 1.0f / omega_light;
    }
};

} // namespace rendertoy3o
