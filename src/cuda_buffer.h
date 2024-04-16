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

    public:
        ~CUDABuffer() {
            RENDERTOY3O_CUDA_CHECK(cuMemFree(_buffer_ptr));
        }
        [[nodiscard]] auto buffer_ptr() const noexcept { return _buffer_ptr; }
    };
}