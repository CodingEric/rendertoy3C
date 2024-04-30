#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <src/util/exception.h>

namespace rendertoy3o
{
    class CUDAStream
    {
    private:
        CUstream _stream{};

    public:
        CUDAStream()
        {
            RENDERTOY3O_CUDA_CHECK(cudaStreamCreate(&_stream));
        }

        void sync()
        {
            RENDERTOY3O_CUDA_CHECK(cudaStreamSynchronize(_stream));
        }

        ~CUDAStream()
        {
            RENDERTOY3O_CUDA_CHECK(cudaStreamDestroy(_stream));
        }

    public:
        const auto &stream() const { return _stream; }
    };
}