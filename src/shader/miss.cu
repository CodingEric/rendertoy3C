#include <src/shader/shader_common.h>

static __forceinline__ __device__ rendertoy3o::RadiancePRD loadMissRadiancePRD()
{
    rendertoy3o::RadiancePRD prd = {};
    return prd;
}

static __forceinline__ __device__ void storeMissRadiancePRD( rendertoy3o::RadiancePRD prd )
{
    optixSetPayload_5( __float_as_uint( prd.emitted.x ) );
    optixSetPayload_6( __float_as_uint( prd.emitted.y ) );
    optixSetPayload_7( __float_as_uint( prd.emitted.z ) );

    optixSetPayload_8( __float_as_uint( prd.radiance.x ) );
    optixSetPayload_9( __float_as_uint( prd.radiance.y ) );
    optixSetPayload_10( __float_as_uint( prd.radiance.z ) );

    optixSetPayload_17( prd.done );
}

extern "C" __global__ void __miss__radiance()
{
    optixSetPayloadTypes( rendertoy3o::PAYLOAD_TYPE_RADIANCE );

    rendertoy3o::MissData* rt_data  = reinterpret_cast<rendertoy3o::MissData*>( optixGetSbtDataPointer() );
    rendertoy3o::RadiancePRD prd = loadMissRadiancePRD();

    // prd.radiance  = make_float3( rt_data->bg_color );
    prd.radiance = optixDirectCall<float3>(0);
    prd.emitted   = make_float3( 0.f );
    prd.done      = true;

    storeMissRadiancePRD( prd );
}