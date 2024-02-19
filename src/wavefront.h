#pragma once

#include <optix.h>
#include <optix_stubs.h>

#include <cuda/helpers.h>
#include <cuda_texture_types.h>

#include <sutil/vec_math.h>

// 这两个宏不应该被编译到 CPU 上
#define WAVEFRONT_GPU __device__
#define WAVEFRONT_GPU_INLINE __device__ __forceinline__

#ifdef __NVCC__
#define WAVEFRONT_CPU_GPU __host__ __device__
#define WAVEFRONT_CPU_GPU_INLINE __host__ __device__ __forceinline__
#else
#define WAVEFRONT_CPU_GPU
#define WAVEFRONT_CPU_GPU_INLINE inline
#endif // __NVCC__

using Spectrum = float3;