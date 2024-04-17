#pragma once

#include <cuda.h>

#include "util/exception.h"

namespace rendertoy3o
{
    template <typename BufferType>
    class CUDABuffer
    {
    private:
        CUdeviceptr _buffer_ptr;
        size_t _buffer_size_in_bytes;

    public:
        CUDABuffer(size_t buffer_size) : _buffer_size_in_bytes{buffer_size * sizeof(BufferType)}
        {
            RENDERTOY3O_CUDA_CHECK(cuMemAlloc(&_buffer_ptr, _buffer_size_in_bytes));
        }
        ~CUDABuffer() {
            RENDERTOY3O_CUDA_CHECK(cuMemFree(_buffer_ptr));
        }
        [[nodiscard]] auto buffer_ptr() const noexcept { return _buffer_ptr; }

    public:
        void copy_from(const void *data)
        {
            RENDERTOY3O_CUDA_CHECK(cuMemcpyHtoD_v2(_buffer_ptr, data, _buffer_size_in_bytes));
        }
    };
}