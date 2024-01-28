#include "shader_common.h"

extern "C" __device__ float3 __direct_callable__test()
{
    return make_float3(0.01f, 0.01f, 0.01f);
}