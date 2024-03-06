#pragma once

#include <src/wavefront.h>
#include <src/util/sampling.h>

#include "random.h"

namespace wavefront {

struct SampledBxDF
{
    Spectrum m_bsdf;
    float3 m_wi;
    float m_p;

    WAVEFRONT_GPU
    SampledBxDF(Spectrum bsdf, float3 wi, float p) : m_bsdf(bsdf), m_wi(wi), m_p(p) {}
};

struct DiffuseBSDF
{
    float3 m_R;

    WAVEFRONT_CPU_GPU
    DiffuseBSDF(const float3 &R, const float3 &T) : m_R(R) {}

    WAVEFRONT_GPU
    Spectrum f(const float3 &wo, const float3 &wi) const
    {
        return m_R * M_1_PIf;
    }

    WAVEFRONT_GPU
    SampledBxDF Sample_f(const float3 &wo, unsigned int &seed) const
    {
        float3 wi = SampleCosineHemisphere(rnd2(seed));
        if (wo.z < 0)
            wi.z *= -1;
        float pdf = SampleCosineHemispherePDF(fabsf(wi.z));
        return SampledBxDF(m_R * M_1_PIf, wi, pdf);
    }

    WAVEFRONT_GPU
    float PDF(const float3 &wo, const float3 &wi) const
    {
        return SampleCosineHemispherePDF(fabsf(wi.z));
    }
};

} // namespace wavefront
