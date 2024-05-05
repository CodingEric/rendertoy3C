#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <src/util/exception.h>

#include <iostream>

#include "cuda_stream.h"

namespace rendertoy3o
{
    template <typename BufferType>
    class CUDABuffer
    {
    private:
        CUdeviceptr _buffer_ptr;
        size_t _buffer_size_in_bytes;

    public:
        CUDABuffer(const CUDABuffer &) = delete;
        CUDABuffer(CUDABuffer &&other) noexcept : _buffer_ptr{other._buffer_ptr},
                                         _buffer_size_in_bytes{other._buffer_size_in_bytes}
        {
            other._buffer_ptr = 0u;
        }
        CUDABuffer(const CUDAStream &stream, size_t buffer_size) noexcept : _buffer_size_in_bytes{buffer_size * sizeof(BufferType)}
        {
            RENDERTOY3O_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&_buffer_ptr), _buffer_size_in_bytes, stream.stream()));
        }
        ~CUDABuffer() noexcept
        {
            if (_buffer_ptr)
            {
                RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(_buffer_ptr)));
            }
        }
        [[nodiscard]] const auto &buffer_ptr() const noexcept { return _buffer_ptr; }

    public:
        void copy_from(const CUDAStream &stream, const void *data) noexcept
        {
            RENDERTOY3O_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(_buffer_ptr), data, _buffer_size_in_bytes, cudaMemcpyHostToDevice, stream.stream()));
        }

        void copy_from(const CUDAStream &stream, const void *data, const size_t offset, const size_t size) noexcept
        {
            RENDERTOY3O_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(_buffer_ptr + offset * sizeof(BufferType)), data, size * sizeof(BufferType), cudaMemcpyHostToDevice, stream.stream()));
        }
    };
}