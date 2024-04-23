#pragma once

#include <src/wavefront.h>

#ifndef M_PIf
#define M_PIf       3.14159265358979323846f
#endif
#ifndef M_PI_2f
#define M_PI_2f     1.57079632679489661923f
#endif
#ifndef M_PI_4f
#define M_PI_4f     0.78539816339744830962f
#endif
#ifndef M_1_PIf
#define M_1_PIf     0.318309886183790671538f
#endif

namespace rendertoy3o {

RENDERTOY_CPU_GPU_INLINE
float safeSqrt(float x) {
    return sqrtf(fmaxf(0.0f, x));
}
// pbrt-v4 为 safeSqrt 引入了 double 重载，但是在 wavefront 中这是不必要的。

} // namespace wavefront