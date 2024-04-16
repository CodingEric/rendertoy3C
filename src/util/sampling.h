#pragma once

#include <src/wavefront.h>
#include <src/util/math.h>

namespace rendertoy3o {

WAVEFRONT_CPU_GPU_INLINE
float2 SampleUniformDiskConcentric(float2 u) {
    // Map _u_ to $[-1,1]^2$ and handle degeneracy at the origin
    float2 uOffset = 2 * u - make_float2(1, 1);
    if (uOffset.x == 0 && uOffset.y == 0)
        return {0, 0};

    // Apply concentric mapping to point
    float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = M_PI_4f * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = M_PI_2f - M_PI_4f * (uOffset.x / uOffset.y);
    }
    return r * make_float2(std::cos(theta), std::sin(theta));
}

WAVEFRONT_CPU_GPU_INLINE
float3 SampleCosineHemisphere(float2 u) {
    // Uniformly sample disk.
    const float r = sqrtf(u.x);
    const float phi = 2.0f * M_PIf * u.y;
    float3 p = {};
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);
    p.z = safeSqrt(1.0f - p.x * p.x - p.y * p.y);
    return p;
}

WAVEFRONT_CPU_GPU_INLINE
float SampleCosineHemispherePDF(float cosTheta) {
    return cosTheta * M_1_PIf;
}

} // namespace wavefront