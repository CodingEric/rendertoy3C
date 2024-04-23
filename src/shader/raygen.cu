#include <src/shader/shader_common.h>

extern "C"
{
    __constant__ rendertoy3o::RenderSettings params;
}

//------------------------------------------------------------------------------
//
// Programs
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__rg()
{
    const int w = params.film_settings.width;
    const int h = params.film_settings.height;
    const float3 eye = params.camera_settings.eye;
    const float3 U = params.camera_settings.U;
    const float3 V = params.camera_settings.V;
    const float3 W = params.camera_settings.W;
    const uint3 idx = optixGetLaunchIndex();
    const int subframe_index = params.film_settings.subframe_index;

    unsigned int seed = tea<4>(idx.y * w + idx.x, subframe_index);

    float3 result = make_float3(0.0f);
    int i = params.film_settings.samples_per_launch;
    do
    {
        // The center of each pixel is at fraction (0.5,0.5)
        const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

        const float2 d = 2.0f * make_float2(
                                    (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(w),
                                    (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(h)) -
                         1.0f;
        float3 ray_direction = normalize(d.x * U + d.y * V + W);
        float3 ray_origin = eye;

        rendertoy3o::RadiancePRD prd;
        prd.attenuation = make_float3(1.f);
        prd.seed = seed;
        prd.depth = 0;

        float3 last_attenuation = prd.attenuation;

        for (;;)
        {
            rendertoy3o::traceRadiance(
                params.handle,
                ray_origin,
                ray_direction,
                0.01f, // tmin       // TODO: smarter offset
                1e16f, // tmax
                prd);

            result += prd.emitted;
            result += prd.radiance * last_attenuation;
            last_attenuation = prd.attenuation;

            const float p = dot(prd.attenuation, make_float3(0.30f, 0.59f, 0.11f));
            const bool done = prd.done || rnd(prd.seed) > p;
            if (done)
                break;
            prd.attenuation /= p;

            ray_origin = prd.origin;
            ray_direction = prd.direction;

            ++prd.depth;
        }
    } while (--i);

    const uint3 launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.film_settings.width + launch_index.x;
    float3 accum_color = result / static_cast<float>(params.film_settings.samples_per_launch);

    if (subframe_index > 0)
    {
        const float a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(params.film_settings.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.film_settings.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
    params.film_settings.frame_buffer[image_index] = make_color(accum_color);
}
