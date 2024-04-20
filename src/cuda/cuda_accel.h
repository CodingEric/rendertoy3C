#pragma once

#include <cuda.h>
#include <optix.h>
#include <vector>
#include <cstring>

#include <src/util/exception.h>
#include "cuda_mesh.h"

namespace rendertoy3o
{
    class CUDAAccel
    {
    private:
        OptixTraversableHandle _ias_handle{0};
        CUdeviceptr _ias_output_buffer{0};
        std::vector<OptixInstance> _optix_instances{};

    public:
        CUDAAccel() = default;
        CUDAAccel(const CUDAAccel &) = delete;
        CUDAAccel(const CUDAAccel &&) = delete;
        ~CUDAAccel()
        {
            RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_ias_output_buffer)));
        }

        void append_instance(const CUDAMesh &mesh, float transformation[12])
        {
            OptixInstance optix_instance{
                .instanceId = static_cast<uint>(instance_size()),
                .sbtOffset = static_cast<uint>(instance_size()),
                .visibilityMask = 1,
                .flags = OPTIX_INSTANCE_FLAG_NONE,
                .traversableHandle = mesh.gas_handle()};
            memcpy(optix_instance.transform, transformation, sizeof(float) * 12);
            _optix_instances.push_back(optix_instance);
        }

        void build(const OptixDeviceContext ctx)
        {
            CUdeviceptr d_instances;
            const size_t instance_size_in_bytes = sizeof(OptixInstance);
            RENDERTOY3O_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_instances), instance_size_in_bytes * instance_size()));
            RENDERTOY3O_CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(d_instances),
                _optix_instances.data(),
                instance_size_in_bytes * instance_size(),
                cudaMemcpyHostToDevice));

            OptixBuildInput instance_input = {};
            instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            instance_input.instanceArray.instances = d_instances;
            instance_input.instanceArray.numInstances = instance_size();

            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
            accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

            // accel_options.motionOptions.numKeys   = 2;
            // accel_options.motionOptions.timeBegin = 0.0f;
            // accel_options.motionOptions.timeEnd   = 1.0f;
            // accel_options.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;

            OptixAccelBufferSizes ias_buffer_sizes;
            RENDERTOY3O_OPTIX_CHECK(optixAccelComputeMemoryUsage(
                ctx,
                &accel_options,
                &instance_input,
                1, // num build inputs
                &ias_buffer_sizes));

            CUdeviceptr d_temp_buffer;
            RENDERTOY3O_CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&d_temp_buffer),
                ias_buffer_sizes.tempSizeInBytes));
            RENDERTOY3O_CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&_ias_output_buffer),
                ias_buffer_sizes.outputSizeInBytes));

            RENDERTOY3O_OPTIX_CHECK(optixAccelBuild(
                ctx,
                0, // CUDA stream
                &accel_options,
                &instance_input,
                1, // num build inputs
                d_temp_buffer,
                ias_buffer_sizes.tempSizeInBytes,
                _ias_output_buffer,
                ias_buffer_sizes.outputSizeInBytes,
                &_ias_handle,
                nullptr, // emitted property list
                0        // num emitted properties
                ));

            RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer)));
            RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_instances)));
        }

    public:
        [[nodiscard]] const size_t instance_size() const
        {
            return _optix_instances.size();
        }

        [[nodiscard]] const auto &ias_handle() const
        {
            return _ias_handle;
        }
    };
}