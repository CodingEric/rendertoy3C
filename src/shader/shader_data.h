#pragma once

#include <src/wavefront.h>
#include <src/light.h>
#include <src/util/type.h>
#include <src/util/exception.h>
#include "random.h"

namespace rendertoy3o
{

    //------------------------------------------------------------------------------
    //
    // Payload
    //
    //------------------------------------------------------------------------------

    constexpr unsigned int RAY_TYPE_COUNT = 1;

    constexpr OptixPayloadTypeID PAYLOAD_TYPE_RADIANCE = OPTIX_PAYLOAD_TYPE_ID_0;

    struct RadiancePRD
    {
        // these are produced by the caller, passed into trace, consumed/modified by CH and MS and consumed again by the caller after trace returned.
        float3 attenuation;
        unsigned int seed;
        int depth;

        // these are produced by CH and MS, and consumed by the caller after trace returned.
        float3 emitted;
        float3 radiance;
        float3 origin;
        float3 direction;
        int done;

        float pdf_prev;
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

    struct RenderSettings
    {
        struct
        {
            unsigned int subframe_index;
            float4 *accum_buffer;
            uchar4 *frame_buffer;
            unsigned int width;
            unsigned int height;
            unsigned int samples_per_launch;
        } film_settings;

        struct
        {
            float3 eye;
            float3 U;
            float3 V;
            float3 W;
        } camera_settings;

        struct
        {
            unsigned int light_count;
            Light *lights;
        } light_settings;

        OptixTraversableHandle handle;

#ifndef __NVCC__
        RenderSettings(int width, int height, uint samples_per_launch, OptixTraversableHandle handle)
        {
            film_settings.width = width;
            film_settings.height = height;
            RENDERTOY3O_CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&film_settings.accum_buffer),
                width * height * sizeof(float4)));
            film_settings.frame_buffer = nullptr; // Will be set when output buffer is mapped

            film_settings.samples_per_launch = samples_per_launch;
            film_settings.subframe_index = 0u;
            this->handle = handle;
        }
#endif // __NVCC__
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
        float3 emission_color;
        float3 diffuse_color;
        float3 *vertices;
        int3 *indices;
        float3 *normals;
        float2 *texcoords;

        bool hasTexture;
        cudaTextureObject_t texture;
    };

    struct CallableData
    {
    };

    template <typename T>
    struct Record
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };
    using RayGenRecord = Record<rendertoy3o::RayGenData>;
    using MissRecord = Record<rendertoy3o::MissData>;
    using HitGroupRecord = Record<rendertoy3o::HitGroupData>;
    using CallableRecord = Record<rendertoy3o::CallableData>;

} // namespace wavefront
