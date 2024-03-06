#include <src/shader/shader_common.h>
#include <src/light.h>
#include <src/util/sampling.h>

extern "C"
{
    __constant__ wavefront::Params params;
}

static __forceinline__ __device__ void SampleLights(const float3 &P, unsigned int &seed, float3 &light_pos, float3 &emission, float &pdf)
{
    wavefront::Light light = params.lights[int(rnd(seed) * params.light_count)];
    light.Sample(P, seed, light_pos, emission, pdf);
    pdf /= params.light_count;
}

static __forceinline__ __device__ wavefront::RadiancePRD loadClosesthitRadiancePRD()
{
    wavefront::RadiancePRD prd = {};

    prd.attenuation.x = __uint_as_float(optixGetPayload_0());
    prd.attenuation.y = __uint_as_float(optixGetPayload_1());
    prd.attenuation.z = __uint_as_float(optixGetPayload_2());
    prd.seed = optixGetPayload_3();
    prd.depth = optixGetPayload_4();
    prd.pdf_prev = __uint_as_float(optixGetPayload_18());
    return prd;
}

static __forceinline__ __device__ void storeClosesthitRadiancePRD(wavefront::RadiancePRD prd)
{
    optixSetPayload_0(__float_as_uint(prd.attenuation.x));
    optixSetPayload_1(__float_as_uint(prd.attenuation.y));
    optixSetPayload_2(__float_as_uint(prd.attenuation.z));

    optixSetPayload_3(prd.seed);
    optixSetPayload_4(prd.depth);

    optixSetPayload_5(__float_as_uint(prd.emitted.x));
    optixSetPayload_6(__float_as_uint(prd.emitted.y));
    optixSetPayload_7(__float_as_uint(prd.emitted.z));

    optixSetPayload_8(__float_as_uint(prd.radiance.x));
    optixSetPayload_9(__float_as_uint(prd.radiance.y));
    optixSetPayload_10(__float_as_uint(prd.radiance.z));

    optixSetPayload_11(__float_as_uint(prd.origin.x));
    optixSetPayload_12(__float_as_uint(prd.origin.y));
    optixSetPayload_13(__float_as_uint(prd.origin.z));

    optixSetPayload_14(__float_as_uint(prd.direction.x));
    optixSetPayload_15(__float_as_uint(prd.direction.y));
    optixSetPayload_16(__float_as_uint(prd.direction.z));

    optixSetPayload_17(prd.done);

    optixSetPayload_18(__float_as_uint(prd.pdf_prev));
}

extern "C" __global__ void __closesthit__radiance()
{
    optixSetPayloadTypes(wavefront::PAYLOAD_TYPE_RADIANCE);

    wavefront::HitGroupData *rt_data = (wavefront::HitGroupData *)optixGetSbtDataPointer();

    const int prim_idx = optixGetPrimitiveIndex();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;
    const int3 index = rt_data->indices[prim_idx];
    float3 Ng = (1.0f - u - v) * rt_data->normals[index.x] + u * rt_data->normals[index.y] + v * rt_data->normals[index.z];
    Ng = normalize(Ng);
    float2 texcoord = (1.0f - u - v) * rt_data->texcoords[index.x] + u * rt_data->texcoords[index.y] + v * rt_data->texcoords[index.z];
    const float3 Ns = faceforward(Ng, -ray_dir, Ng);
    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
    wavefront::RadiancePRD prd = loadClosesthitRadiancePRD();

    if (prd.depth == 0)
        prd.emitted = rt_data->emission_color;
    else
    {
        prd.emitted = make_float3(0.0f);
    }

    unsigned int seed = prd.seed;

    // ---------------------------------------------
    // BSDF Sampling
    // ---------------------------------------------
    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 w_in = wavefront::SampleCosineHemisphere(rnd2(seed));
        prd.pdf_prev = w_in.z / M_PI;
        wavefront::Onb onb(Ns);
        onb.inverse_transform(w_in);
        prd.direction = w_in;
        prd.origin = P;

        float bsdf = 1.0f / M_PI;

        if (rt_data->hasTexture)
        {
            prd.attenuation *= make_float3(tex2D<float4>(rt_data->texture, texcoord.x, texcoord.y));
        }
        else
        {
            prd.attenuation *= rt_data->diffuse_color;
        }
        prd.attenuation *= bsdf / prd.pdf_prev;
    }

    // ---------------------------------------------
    // Next Event Estimation
    // ---------------------------------------------
    {
        float3 light_pos;
        float3 light_emission;
        float pdf_light;
        SampleLights(P, seed, light_pos, light_emission, pdf_light);

        prd.seed = seed;

        const float Ldist = length(light_pos - P);
        const float3 L = normalize(light_pos - P);
        const float nDl = dot(Ns, L);

        float3 weight = {0.0f, 0.0f, 0.0f};
        if (nDl > 0.0f)
        {
            const bool occluded =
                wavefront::traceOcclusion(
                    params.handle,
                    P,
                    L,
                    0.001f,         // tmin
                    Ldist - 0.01f); // tmax

            if (!occluded)
            {
                float pdf_scattering = abs(dot(L, Ns)) / M_PI;
                float bsdf = 1.0f / M_PI;
                if (rt_data->hasTexture)
                {
                    weight = make_float3(tex2D<float4>(rt_data->texture, texcoord.x, texcoord.y));
                }
                else
                {
                    weight = rt_data->diffuse_color;
                }
                weight *= wavefront::powerHeuristic(pdf_light, pdf_scattering) * bsdf;
            }
        }
        prd.radiance = light_emission * weight;
    }
    prd.done = false;

    storeClosesthitRadiancePRD(prd);
}