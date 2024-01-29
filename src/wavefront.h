#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sutil/vec_math.h>

#if __NVCC__
#define WAVEFRONT_CPU_GPU __host__ __device__
#define WAVEFRONT_GPU __device__ __forceinline__
#else
#define WAVEFRONT_CPU_GPU
#define WAVEFRONT_GPU
#endif // __NVCC__