#pragma once

#include <optix.h>
#include <optix_stubs.h>

#include <cuda/helpers.h>
#include <cuda_texture_types.h>

#include <sutil/vec_math.h>

// 这两个宏不应该被编译到 CPU 上，不在 __NVCC__ 下启用的目的是让编译器报错。
#define RENDERTOY_GPU __device__
#define RENDERTOY_GPU_INLINE __device__ __forceinline__

#define RENDERTOY_CPU __host__

#ifdef __NVCC__
#define RENDERTOY_CPU_GPU __host__ __device__
#define RENDERTOY_CPU_GPU_INLINE __host__ __device__ __forceinline__
#else
#define RENDERTOY_CPU_GPU
#define RENDERTOY_CPU_GPU_INLINE inline
#endif // __NVCC__

namespace rendertoy3o
{

    using Spectrum = float3;

    // Forward declarations
    struct Mesh;
    struct Texture;
}