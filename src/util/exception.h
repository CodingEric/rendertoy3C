#pragma once

#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include <optix.h>
#include <stdexcept>
#include <sstream>

#define RENDERTOY3O_CUDA_CHECK(call) ::rendertoy3o::cudaCheck(call, #call, __FILE__, __LINE__)
#define RENDERTOY3O_OPTIX_CHECK(call) ::rendertoy3o::optixCheck(call, #call, __FILE__, __LINE__)
#define RENDERTOY3O_OPTIX_CHECK_LOG(call)                                     \
    do                                                                        \
    {                                                                         \
        char LOG[2048];                                                       \
        size_t LOG_SIZE = sizeof(LOG);                                        \
        ::rendertoy3o::optixCheckLog(call, LOG, sizeof(LOG), LOG_SIZE, #call, \
                                     __FILE__, __LINE__);                     \
    } while (false)

namespace rendertoy3o
{

    class Exception : public std::runtime_error
    {
    public:
        Exception(const char *msg)
            : std::runtime_error(msg)
        {
        }

        Exception(OptixResult res, const char *msg)
            : std::runtime_error(createMessage(res, msg).c_str())
        {
        }

    private:
        std::string createMessage(OptixResult res, const char *msg)
        {
            std::ostringstream out;
            out << optixGetErrorName(res) << ": " << msg;
            return out.str();
        }
    };

    inline void cudaCheck(cudaError_t error, const char *call, const char *file, unsigned int line)
    {
        if (error != cudaSuccess)
        {
            std::stringstream ss;
            ss << "CUDA call (" << call << " ) failed with error: '"
               << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";
            throw Exception(ss.str().c_str());
        }
    }

    inline void optixCheck(OptixResult res, const char *call, const char *file, unsigned int line)
    {
        if (res != OPTIX_SUCCESS)
        {
            std::stringstream ss;
            ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\n";
            throw Exception(res, ss.str().c_str());
        }
    }

    inline void optixCheckLog(OptixResult res,
                              const char *log,
                              size_t sizeof_log,
                              size_t sizeof_log_returned,
                              const char *call,
                              const char *file,
                              unsigned int line)
    {
        if (res != OPTIX_SUCCESS)
        {
            std::stringstream ss;
            ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\nLog:\n"
               << log << (sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "") << '\n';
            throw Exception(res, ss.str().c_str());
        }
    }
}