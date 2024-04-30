#pragma once

#include "cuda_accel.h"
#include "cuda_mesh.h"
#include "cuda_texture.h"
#include "optix_context.h"
#include "cuda_stream.h"

#include <src/shader/shader_data.h> // TODO: is it necessary?

namespace rendertoy3o
{
    class CUDAScene
    {
    private:
        const OptixContext &_optix_context;

        std::vector<CUDAMesh> _cuda_meshes = {};
        std::vector<CUDATexture<uchar4>> _cuda_textures = {};
        OptixShaderBindingTable _sbt = {};
        CUDAAccel _accel;

        CUdeviceptr _cuda_params{0u};

    private:
        void create_sbt(const CUDAStream &stream, const std::vector<Mesh> &meshes)
        {
            CUdeviceptr d_raygen_record;
            const size_t raygen_record_size = sizeof(RayGenRecord);
            RENDERTOY3O_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&d_raygen_record), raygen_record_size, stream.stream()));

            RayGenRecord rg_sbt = {};
            RENDERTOY3O_OPTIX_CHECK(optixSbtRecordPackHeader(_optix_context.raygen_prog_group(), &rg_sbt));

            RENDERTOY3O_CUDA_CHECK(cudaMemcpyAsync(
                reinterpret_cast<void *>(d_raygen_record),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice, 
                stream.stream()));

            CUdeviceptr d_miss_records;
            const size_t miss_record_size = sizeof(MissRecord);
            RENDERTOY3O_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&d_miss_records), miss_record_size * rendertoy3o::RAY_TYPE_COUNT, stream.stream()));

            MissRecord ms_sbt[1];
            RENDERTOY3O_OPTIX_CHECK(optixSbtRecordPackHeader(_optix_context.radiance_miss_group(), &ms_sbt[0]));
            ms_sbt[0].data.bg_color = make_float4(0.0f);

            RENDERTOY3O_CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(d_miss_records),
                ms_sbt,
                miss_record_size * rendertoy3o::RAY_TYPE_COUNT,
                cudaMemcpyHostToDevice));

            CUdeviceptr d_hitgroup_records;
            const size_t hitgroup_record_size = sizeof(HitGroupRecord);
            RENDERTOY3O_CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void **>(&d_hitgroup_records),
                hitgroup_record_size * meshes.size(), stream.stream()));

            std::vector<HitGroupRecord> hitGroupRecords;
            for (size_t i = 0; i < meshes.size(); ++i)
            {
                HitGroupRecord record;
                RENDERTOY3O_OPTIX_CHECK(optixSbtRecordPackHeader(_optix_context.radiance_hit_group(), &record));
                // record.data.diffuse_color = {0.8f, 0.8f, 0.8f};
                if (meshes[i].material.m_diffuseTextureID != -1)
                {
                    record.data.hasTexture = true;
                    record.data.texture = _cuda_textures[meshes[i].material.m_diffuseTextureID].texture_object();
                }
                else
                {
                    record.data.hasTexture = false;
                    record.data.diffuse_color = meshes[i].material.m_diffuse;
                }
                record.data.emission_color = meshes[i].material.m_emissive;
                record.data.vertices = reinterpret_cast<float3 *>(_cuda_meshes[i].vertex_buffer().buffer_ptr());
                record.data.indices = reinterpret_cast<int3 *>(_cuda_meshes[i].index_buffer().buffer_ptr());
                record.data.normals = reinterpret_cast<float3 *>(_cuda_meshes[i].normal_buffer().buffer_ptr());
                record.data.texcoords = reinterpret_cast<float2 *>(_cuda_meshes[i].texcoord_buffer().buffer_ptr());
                hitGroupRecords.push_back(record);
            }

            RENDERTOY3O_CUDA_CHECK(cudaMemcpyAsync(
                reinterpret_cast<void *>(d_hitgroup_records),
                hitGroupRecords.data(),
                hitgroup_record_size * meshes.size(),
                cudaMemcpyHostToDevice, 
                stream.stream()));

            CUdeviceptr d_callable_records;
            const size_t callable_record_size = sizeof(CallableRecord);
            RENDERTOY3O_CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void **>(&d_callable_records),
                callable_record_size * 1, 
                stream.stream()));
            std::vector<CallableRecord> callableRecords;
            for (size_t i = 0; i < 1; ++i)
            {
                CallableRecord record;
                RENDERTOY3O_OPTIX_CHECK(optixSbtRecordPackHeader(_optix_context.callable_test_group(), &record));
                callableRecords.push_back(record);
            }

            RENDERTOY3O_CUDA_CHECK(cudaMemcpyAsync(
                reinterpret_cast<void *>(d_callable_records),
                callableRecords.data(),
                callable_record_size * 1,
                cudaMemcpyHostToDevice, 
                stream.stream()));

            _sbt.raygenRecord = d_raygen_record;
            _sbt.missRecordBase = d_miss_records;
            _sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
            _sbt.missRecordCount = rendertoy3o::RAY_TYPE_COUNT;
            _sbt.hitgroupRecordBase = d_hitgroup_records;
            _sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
            _sbt.hitgroupRecordCount = meshes.size();
            _sbt.callablesRecordBase = d_callable_records;
            _sbt.callablesRecordCount = 1;
            _sbt.callablesRecordStrideInBytes = static_cast<uint32_t>(callable_record_size);
        }

    public:
        CUDAScene(const CUDAScene &) = delete;
        CUDAScene(CUDAScene &&) = delete;
        CUDAScene(const CUDAStream &stream, const OptixContext &optix_context, const std::vector<Mesh> &meshes, const std::vector<Texture> &textures)
        : _optix_context(optix_context)
        {
            RENDERTOY3O_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&_cuda_params), sizeof(rendertoy3o::RenderSettings), stream.stream()));

            for (const auto &mesh : meshes)
            {
                _cuda_meshes.push_back(CUDAMesh(stream, optix_context.ctx(), mesh));
            }

            // const std::vector<std::array<float, 12>> motion_matrix =
            //     {{1.0f, 0.0f, 0.0f, 0.0f,
            //       0.0f, 1.0f, 0.0f, 0.0f,
            //       0.0f, 0.0f, 1.0f, 0.0f},
            //      {1.0f, 0.0f, 0.0f, 0.0f,
            //       0.0f, 1.0f, 0.0f, 0.5f,
            //       0.0f, 0.0f, 1.0f, 0.0f}};

            float transformation[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
            for (const auto &mesh : _cuda_meshes)
            {
                _accel.append_instance(mesh, transformation);
                // _accel.append_animated_instance(optix_context.ctx(), mesh, motion_matrix, OptixMotionOptions {.flags = OPTIX_MOTION_FLAG_NONE, .numKeys = 2u, .timeBegin = 0.f, .timeEnd = 1.f}, transformation);
            }
            _accel.build(optix_context.ctx());

            for (const auto &texture : textures)
            {
                _cuda_textures.push_back(CUDATexture<uchar4>(texture.resolution.x,
                                                             texture.resolution.y,
                                                             texture.pixel.data(),
                                                             CUDATexture<uchar4>::AddressMode::Wrap,
                                                             CUDATexture<uchar4>::FilterMode::Linear));
            }

            create_sbt(stream, meshes);
        }

        ~CUDAScene()
        {
            RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_sbt.raygenRecord)));
            RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_sbt.missRecordBase)));
            RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_sbt.hitgroupRecordBase)));
            RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_sbt.callablesRecordBase)));

            RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_cuda_params)));
        }

    public:
        void update_cuda_params_async(const RenderSettings &params, const cudaStream_t stream) const
        {
            RENDERTOY3O_CUDA_CHECK(cudaMemcpyAsync(
                reinterpret_cast<void *>(_cuda_params),
                &params, sizeof(rendertoy3o::RenderSettings),
                cudaMemcpyHostToDevice, stream));
        }

    public:
        [[nodiscard]] const auto &sbt() const { return _sbt; }
        [[nodiscard]] const auto &accel() const { return _accel; }
        [[nodiscard]] const auto &params() const { return _cuda_params; }
    };
}