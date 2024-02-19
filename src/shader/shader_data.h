#pragma once

#include <src/wavefront.h>
#include <src/light.h>
#include "random.h"

//------------------------------------------------------------------------------
//
// Payload
//
//------------------------------------------------------------------------------

constexpr unsigned int RAY_TYPE_COUNT = 1;

constexpr OptixPayloadTypeID PAYLOAD_TYPE_RADIANCE  = OPTIX_PAYLOAD_TYPE_ID_0;

struct RadiancePRD
{
    // these are produced by the caller, passed into trace, consumed/modified by CH and MS and consumed again by the caller after trace returned.
    float3       attenuation;
    unsigned int seed;
    int          depth;

    // these are produced by CH and MS, and consumed by the caller after trace returned.
    float3       emitted;
    float3       radiance;
    float3       origin;
    float3       direction;
    int          done;

    float        pdf_prev;
};


const unsigned int radiancePayloadSemantics[19] =
{
    // RadiancePRD::attenuation
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // RadiancePRD::seed
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // RadiancePRD::depth
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // RadiancePRD::emitted
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    // RadiancePRD::radiance
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    // RadiancePRD::origin
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    // RadiancePRD::direction
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    // RadiancePRD::done
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    // RadiancePRD::pdf_prev
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
};


struct Params
{
    unsigned int subframe_index;
    float4*      accum_buffer;
    uchar4*      frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;

    unsigned int light_count;
    Light *lights;
    OptixTraversableHandle handle;
};


struct RayGenData
{
};


struct MissData
{
    float4 bg_color;
};


struct HitGroupData
{
    float3  emission_color;
    float3  diffuse_color;
    float3* vertices;
    int3* indices;
    float3* normals;
    float2* texcoords;

    bool hasTexture;
    cudaTextureObject_t texture;
};

struct CallableData
{

};

