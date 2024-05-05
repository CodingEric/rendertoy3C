#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <src/util/exception.h>

#include <type_traits>

namespace rendertoy3o
{
    template <typename BufferType>
    class CUDATexture
    {
    public:
        enum struct AddressMode : int32_t
        {
            Wrap = 0,
            Clamp = 1,
            Mirror = 2,
            Border = 3
        };

        enum struct FilterMode : int32_t
        {
            Linear = 0,
            Point = 1
        };

    private:
        cudaArray_t _texture_array{0u};
        cudaTextureObject_t _texture_object{0u};

        size_t _width, _height;

    public:
        CUDATexture(const CUDATexture &) = delete;
        CUDATexture(CUDATexture &&other) noexcept : _texture_array{other._texture_array},
                                           _texture_object{other._texture_object},
                                           _width{other._width},
                                           _height{other._height}
        {
            other._texture_array = 0u;
            other._texture_object = 0u;
        }
        CUDATexture(size_t width,
                    size_t height,
                    const void *data,
                    const AddressMode address_mode,
                    const FilterMode filter_mode) noexcept : _width{width}, _height{height}
        {
            cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<BufferType>();
            RENDERTOY3O_CUDA_CHECK(cudaMallocArray(&_texture_array, &channel_desc, width, height));

            size_t pitch = _width * sizeof(BufferType);
            RENDERTOY3O_CUDA_CHECK(cudaMemcpy2DToArray(_texture_array, 0, 0, data, pitch, pitch, _height, cudaMemcpyHostToDevice));

            cudaResourceDesc res_desc = {};
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = _texture_array;

            cudaTextureDesc tex_desc = {};
            tex_desc.addressMode[0] = static_cast<cudaTextureAddressMode>(address_mode);
            tex_desc.addressMode[1] = static_cast<cudaTextureAddressMode>(address_mode);
            tex_desc.filterMode = static_cast<cudaTextureFilterMode>(filter_mode);
            tex_desc.readMode = cudaReadModeNormalizedFloat;
            tex_desc.normalizedCoords = 1;
            tex_desc.maxAnisotropy = 1;
            tex_desc.maxMipmapLevelClamp = 99;
            tex_desc.minMipmapLevelClamp = 0;
            tex_desc.mipmapFilterMode = cudaFilterModePoint;
            tex_desc.sRGB = 0;

            RENDERTOY3O_CUDA_CHECK(cudaCreateTextureObject(&_texture_object, &res_desc, &tex_desc, nullptr));
        }

        ~CUDATexture() noexcept
        {
            if (_texture_object)
                RENDERTOY3O_CUDA_CHECK(cudaDestroyTextureObject(_texture_object));
            if (_texture_array)
                RENDERTOY3O_CUDA_CHECK(cudaFreeArray(_texture_array));
        }

    public:
        const auto texture_object() const noexcept
        {
            return _texture_object;
        }
    };
}