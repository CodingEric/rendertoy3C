#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "util/exception.h"

namespace rendertoy3o
{
    class CUDATexture
    {
    private:
        cudaTextureObject_t cudaTex{0};

    public:
        ~CUDATexture(){
            RENDERTOY3O_CUDA_CHECK(cudaDestroyTextureObject(cudaTex));
        }
    };
}