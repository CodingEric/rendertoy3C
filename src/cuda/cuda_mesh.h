#pragma once

#include "cuda_buffer.h"
#include <src/mesh.h>

namespace rendertoy3o
{
    class CUDAMesh
    {
    private:
        CUDABuffer<float3> _vertex_buffer;
        CUDABuffer<int3> _index_buffer;
        CUDABuffer<float3> _normal_buffer;
        CUDABuffer<float2> _texcoord_buffer;

        std::vector<CUdeviceptr> _vertices_per_key;
        OptixTraversableHandle _gas_handle;
        CUdeviceptr _gas_output_buffer;

    public:
        CUDAMesh(const CUDAMesh &) = delete;
        CUDAMesh(CUDAMesh &&other) : _vertex_buffer{std::move(other._vertex_buffer)},
                                     _index_buffer{std::move(other._index_buffer)},
                                     _normal_buffer{std::move(other._normal_buffer)},
                                     _texcoord_buffer{std::move(other._texcoord_buffer)},
                                     _vertices_per_key{std::move(other._vertices_per_key)},
                                     _gas_handle{other._gas_handle},
                                     _gas_output_buffer{other._gas_output_buffer}
        {
            other._gas_handle = 0u;
            other._gas_output_buffer = 0u;
        }
        CUDAMesh(OptixDeviceContext ctx, const Mesh &mesh) : _vertex_buffer{CUDABuffer<float3>(mesh.vertices[0].size())}, // TODO: more explicit representation
                                                             _index_buffer{CUDABuffer<int3>(mesh.indices.size())},
                                                             _normal_buffer{CUDABuffer<float3>(mesh.normals[0].size())},
                                                             _texcoord_buffer{CUDABuffer<float2>(mesh.texcoords[0].size())},
                                                             _vertices_per_key{mesh.num_keys}
        {
            for (unsigned int j = 0; j < mesh.num_keys; ++j)
            {
                _vertex_buffer.copy_from(&mesh.vertices[j][0], j * mesh.vertices[0].size(), mesh.vertices[0].size());
                const size_t per_keyframe_vertices_size_in_bytes = mesh.vertices[0].size() * sizeof(float3);

                for (unsigned int j = 0; j < mesh.num_keys; ++j)
                {
                    _vertices_per_key[j] = _vertex_buffer.buffer_ptr() + j * per_keyframe_vertices_size_in_bytes;
                }

                _index_buffer.copy_from(mesh.indices.data(), 0u, mesh.indices.size());
                for (unsigned int j = 0; j < mesh.num_keys; ++j)
                {
                    _normal_buffer.copy_from(&mesh.normals[j][0], j * mesh.normals[0].size(), mesh.normals[0].size());
                }
                for (unsigned int j = 0; j < mesh.num_keys; ++j)
                {
                    _texcoord_buffer.copy_from(&mesh.texcoords[j][0], j * mesh.texcoords[0].size(), mesh.texcoords[0].size());
                }

                OptixBuildInput triangleInput{};
                triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

                triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
                triangleInput.triangleArray.numVertices = static_cast<uint32_t>(mesh.vertices[0].size());
                triangleInput.triangleArray.vertexBuffers = _vertices_per_key.data();

                triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                triangleInput.triangleArray.indexStrideInBytes = sizeof(int3);
                triangleInput.triangleArray.numIndexTriplets = (int)mesh.indices.size();
                // triangleInput.triangleArray.indexBuffer = state.d_indices[i];
                triangleInput.triangleArray.indexBuffer = _index_buffer.buffer_ptr();

                // triangleInputFlags[i] = 0;
                uint32_t triangleInputFlag = 0;
                triangleInput.triangleArray.flags = &triangleInputFlag;
                triangleInput.triangleArray.numSbtRecords = 1;
                triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
                triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
                triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

                // 加速结构设置
                OptixAccelBuildOptions accel_options = {};
                accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION; // 这是加速结构压缩必须的第1个要求
                accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
                accel_options.motionOptions.numKeys = mesh.num_keys;
                accel_options.motionOptions.timeBegin = 0.0f;
                accel_options.motionOptions.timeEnd = 1.0f;
                accel_options.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;

                // 计算加速结构在GPU上[将要]占用的显示内存大小。OptixAccelBufferSizes是一个结构体，内部含有三个参量。这里由于没有更新BVH，所以仅仅采用前两个变量。
                OptixAccelBufferSizes gas_buffer_sizes;
                RENDERTOY3O_OPTIX_CHECK(optixAccelComputeMemoryUsage(
                    ctx,
                    &accel_options,
                    &triangleInput,
                    1, // num_build_inputs
                    &gas_buffer_sizes));

                // 在设备上申请临时存储空间。
                CUdeviceptr d_temp_buffer;
                RENDERTOY3O_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

                // non-compacted output
                // 在设备上申请输出内存空间。
                CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
                size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
                RENDERTOY3O_CUDA_CHECK(cudaMalloc(
                    reinterpret_cast<void **>(&d_buffer_temp_output_gas_and_compacted_size),
                    compactedSizeOffset + 8));

                // 配置加速结构构建器以支持大小压缩。
                OptixAccelEmitDesc emitProperty = {};
                emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE; // 这是加速结构压缩必须的第2个要求
                emitProperty.result = (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

                // 执行加速结构构建。
                // 注意在这个构建过程中并没有对BVH进行压缩。
                // 但是这一个过程会计算对BVH压缩以后的大小并且发射到emitProperty中。
                RENDERTOY3O_OPTIX_CHECK(optixAccelBuild(
                    ctx,
                    0, // CUDA stream
                    &accel_options,
                    &triangleInput,
                    1, // num build inputs
                    d_temp_buffer,
                    gas_buffer_sizes.tempSizeInBytes,
                    d_buffer_temp_output_gas_and_compacted_size,
                    gas_buffer_sizes.outputSizeInBytes,
                    &_gas_handle,
                    &emitProperty, // emitted property list
                    1              // num emitted properties
                    ));

                RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer)));
                // RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_mat_indices)));

                size_t compacted_gas_size;
                RENDERTOY3O_CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

                // 这是技术手册要求的在CPU上执行的判断，因为BVH压缩过程可能在极端情况下导致结果变差。
                if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
                {
                    RENDERTOY3O_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&_gas_output_buffer), compacted_gas_size));

                    // use handle as input and output
                    RENDERTOY3O_OPTIX_CHECK(optixAccelCompact(ctx, 0, _gas_handle, _gas_output_buffer, compacted_gas_size, &_gas_handle));

                    RENDERTOY3O_CUDA_CHECK(cudaFree((void *)d_buffer_temp_output_gas_and_compacted_size));
                }
                else
                {
                    _gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
                }
            }
        }
        ~CUDAMesh()
        {
            if(_gas_output_buffer) { RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_gas_output_buffer))); }
        }

    public:
        [[nodiscard]] const auto &gas_handle() const
        {
            return _gas_handle;
        }
        [[nodiscard]] const auto &vertex_buffer() const
        {
            return _vertex_buffer;
        }
        [[nodiscard]] const auto &index_buffer() const
        {
            return _index_buffer;
        }
        [[nodiscard]] const auto &normal_buffer() const
        {
            return _normal_buffer;
        }
        [[nodiscard]] const auto &texcoord_buffer() const
        {
            return _texcoord_buffer;
        }
    };
}