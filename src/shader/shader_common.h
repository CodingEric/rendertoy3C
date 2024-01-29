#pragma once

#include <optix.h>
#include "random.h"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>

#include "shader_data.h"

#include <src/wavefront.h>


//------------------------------------------------------------------------------
//
// Orthonormal basis helper
//
//------------------------------------------------------------------------------

struct Onb
{
    WAVEFRONT_CPU_GPU
    Onb(const float3 &normal)
    {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z))
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else
        {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    WAVEFRONT_CPU_GPU
    void inverse_transform(float3 &p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

WAVEFRONT_CPU_GPU
static void cosine_sample_hemisphere(const float u1, const float u2, float3 &p)
{
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

WAVEFRONT_GPU
static void traceRadiance(
    OptixTraversableHandle handle,
    float3 ray_origin,
    float3 ray_direction,
    float tmin,
    float tmax,
    RadiancePRD &prd)
{
    unsigned int u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18;

    u0 = __float_as_uint(prd.attenuation.x);
    u1 = __float_as_uint(prd.attenuation.y);
    u2 = __float_as_uint(prd.attenuation.z);
    u3 = prd.seed;
    u4 = prd.depth;
    u18 = __float_as_uint(prd.pdf_prev);

    // Note:
    // This demonstrates the usage of the OptiX shader execution reordering
    // (SER) API.  In the case of this computationally simple shading code,
    // there is no real performance benefit.  However, with more complex shaders
    // the potential performance gains offered by reordering are significant.
    optixTraverse(
        PAYLOAD_TYPE_RADIANCE,
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f, // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0,              // SBT offset
        RAY_TYPE_COUNT, // SBT stride
        0,              // missSBTIndex
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18);
    optixReorder(
        // Application specific coherence hints could be passed in here
    );

    optixInvoke(PAYLOAD_TYPE_RADIANCE,
                u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18);

    prd.attenuation = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
    prd.seed = u3;
    prd.depth = u4;

    prd.emitted = make_float3(__uint_as_float(u5), __uint_as_float(u6), __uint_as_float(u7));
    prd.radiance = make_float3(__uint_as_float(u8), __uint_as_float(u9), __uint_as_float(u10));
    prd.origin = make_float3(__uint_as_float(u11), __uint_as_float(u12), __uint_as_float(u13));
    prd.direction = make_float3(__uint_as_float(u14), __uint_as_float(u15), __uint_as_float(u16));
    prd.done = u17;
    prd.pdf_prev = __uint_as_float(u18);
}

// Returns true if ray is occluded, else false
WAVEFRONT_GPU
static bool traceOcclusion(
    OptixTraversableHandle handle,
    float3 ray_origin,
    float3 ray_direction,
    float tmin,
    float tmax)
{
    // We are only casting probe rays so no shader invocation is needed
    optixTraverse(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax, 0.0f, // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,              // SBT offset
        RAY_TYPE_COUNT, // SBT stride
        0               // missSBTIndex
    );
    return optixHitObjectIsHit();
}

WAVEFRONT_CPU_GPU
static float powerHeuristic(
    const float &p1,
    const float &p2
)
{
    const float p1_2 = p1 * p1;
    const float p2_2 = p2 * p2;
    return p1_2 / (p1_2 + p2_2);
}
