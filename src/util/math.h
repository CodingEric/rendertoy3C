#pragma once

#include <src/wavefront.h>

WAVEFRONT_CPU_GPU_INLINE
float safeSqrt(float x) {
    return sqrtf(fmaxf(0.0f, x));
}
// pbrt-v4 为 safeSqrt 引入了 double 重载，但是在 wavefront 中这是不必要的。