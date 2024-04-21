#pragma once

#include <cuda.h>
#include <optix.h>
#include <vector>
#include <array>
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

        std::vector<CUdeviceptr> _motion_transforms{};
        std::vector<OptixTraversableHandle> _animated_gas_handles{};

    public:
        CUDAAccel() = default;
        CUDAAccel(const CUDAAccel &) = delete;
        CUDAAccel(const CUDAAccel &&) = delete;
        ~CUDAAccel()
        {
            RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_ias_output_buffer)));
            for (const auto &motion_transform : _motion_transforms)
            {
                RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(motion_transform)));
            }
        }

        void append_animated_instance(const OptixDeviceContext ctx, const CUDAMesh &mesh, const std::vector<std::array<float, 12>> &motion_matrix, const OptixMotionOptions &motion_options, float static_transformation[12])
        {

            OptixMatrixMotionTransform motion_transform{.child = mesh.gas_handle(),
                                                        .motionOptions = motion_options};
            CUdeviceptr motion_transform_ptr{0u};
            size_t motion_transform_stub_size_in_bytes = sizeof(OptixMatrixMotionTransform) - 2u * 12u * sizeof(float);
            size_t alloc_size_in_bytes = motion_transform_stub_size_in_bytes + motion_matrix.size() * 12u * sizeof(float);
            RENDERTOY3O_CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&motion_transform_ptr),
                alloc_size_in_bytes));

            RENDERTOY3O_CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(motion_transform_ptr),
                &motion_transform,
                motion_transform_stub_size_in_bytes,
                cudaMemcpyHostToDevice));

            RENDERTOY3O_CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(motion_transform_ptr + motion_transform_stub_size_in_bytes),
                motion_matrix.data(),
                motion_matrix.size() * 12u * sizeof(float),
                cudaMemcpyHostToDevice));

            OptixTraversableHandle temp_animated_gas_handle;
            RENDERTOY3O_OPTIX_CHECK(optixConvertPointerToTraversableHandle(
                ctx,
                motion_transform_ptr,
                OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM,
                &temp_animated_gas_handle));

            _motion_transforms.push_back(motion_transform_ptr);
            _animated_gas_handles.push_back(temp_animated_gas_handle);

            append_instance(temp_animated_gas_handle, static_transformation);
        }

        void append_instance(const OptixTraversableHandle handle, float transformation[12])
        {
            OptixInstance optix_instance{
                .instanceId = static_cast<uint>(instance_size()),
                .sbtOffset = static_cast<uint>(instance_size()),
                .visibilityMask = 1,
                .flags = OPTIX_INSTANCE_FLAG_NONE,
                .traversableHandle = handle};
            memcpy(optix_instance.transform, transformation, sizeof(float) * 12);
            _optix_instances.push_back(optix_instance);
        }

        void append_instance(const CUDAMesh &mesh, float transformation[12])
        {
            append_instance(mesh.gas_handle(), transformation);
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