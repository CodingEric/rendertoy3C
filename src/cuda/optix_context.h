#pragma once

#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sutil/sutil.h>

#include <iostream>
#include <iomanip>

#include <src/util/exception.h>

#include <src/shader/shader_data.h> // Is it good?

namespace rendertoy3o
{
    class OptixContext
    {
    private:
        template <typename T>
        struct Record
        {
            __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
            T data;
        };

        typedef Record<rendertoy3o::RayGenData> RayGenRecord;
        typedef Record<rendertoy3o::MissData> MissRecord;
        typedef Record<rendertoy3o::HitGroupData> HitGroupRecord;
        typedef Record<rendertoy3o::CallableData> CallableRecord;

    private:
        OptixDeviceContext _ctx;
        OptixPipelineCompileOptions _pipeline_compile_options = {};

        OptixModule _ptx_module = 0;
        OptixModule _ptx_miss_module = 0;
        OptixModule _ptx_closehit = 0;
        OptixModule _ptx_test = 0;

        OptixProgramGroup _raygen_prog_group = 0;
        OptixProgramGroup _radiance_miss_group = 0;
        OptixProgramGroup _radiance_hit_group = 0;
        OptixProgramGroup _callable_test_group = 0;

        OptixPipeline _pipeline = 0;

    private:
        static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
        {
            std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
        }

        void createModule()
        {
            OptixPayloadType payloadType = {};
            // radiance prd
            // 辐照度payload，这里是文件结构
            payloadType.numPayloadValues = sizeof(rendertoy3o::radiancePayloadSemantics) / sizeof(rendertoy3o::radiancePayloadSemantics[0]);
            payloadType.payloadSemantics = rendertoy3o::radiancePayloadSemantics;

            OptixModuleCompileOptions module_compile_options = {};
#if !defined(NDEBUG)
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
            module_compile_options.numPayloadTypes = 1;
            module_compile_options.payloadTypes = &payloadType;

            _pipeline_compile_options.usesMotionBlur = true; // 运动模糊：需要将这个选项设置为true。
            _pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
            _pipeline_compile_options.numPayloadValues = 0;
            _pipeline_compile_options.numAttributeValues = 2;
            _pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
            _pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
            _pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE; // 根据文档的指示，如果场景中只包含三角形，那么为了最佳性能，应该启用这个flag。
            _pipeline_compile_options.allowOpacityMicromaps = 0;

            size_t inputSize = 0;
            // 一个非常神奇的发现：这个程序的.cu文件似乎是动态加载的！（应该是加载了被nvcc编译后的结果）
            const char *input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "raygen.cu", inputSize);

            RENDERTOY3O_OPTIX_CHECK_LOG(optixModuleCreate(
                _ctx,
                &module_compile_options,
                &_pipeline_compile_options,
                input,
                inputSize,
                LOG, &LOG_SIZE,
                &_ptx_module));

            const char *input_miss = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "miss.cu", inputSize);

            RENDERTOY3O_OPTIX_CHECK_LOG(optixModuleCreate(
                _ctx,
                &module_compile_options,
                &_pipeline_compile_options,
                input_miss,
                inputSize,
                LOG, &LOG_SIZE,
                &_ptx_miss_module));

            const char *input_closehit = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "closehit_radiance.cu", inputSize);

            RENDERTOY3O_OPTIX_CHECK_LOG(optixModuleCreate(
                _ctx,
                &module_compile_options,
                &_pipeline_compile_options,
                input_closehit,
                inputSize,
                LOG, &LOG_SIZE,
                &_ptx_closehit));

            const char *input_test = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "test.cu", inputSize);

            RENDERTOY3O_OPTIX_CHECK_LOG(optixModuleCreate(
                _ctx,
                &module_compile_options,
                &_pipeline_compile_options,
                input_test,
                inputSize,
                LOG, &LOG_SIZE,
                &_ptx_test));
        }

        void createProgramGroups()
        {
            OptixProgramGroupOptions program_group_options = {};

            {
                OptixProgramGroupDesc raygen_prog_group_desc = {};
                raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                raygen_prog_group_desc.raygen.module = _ptx_module;
                raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

                RENDERTOY3O_OPTIX_CHECK_LOG(optixProgramGroupCreate(
                    _ctx, &raygen_prog_group_desc,
                    1, // num program groups
                    &program_group_options,
                    LOG, &LOG_SIZE,
                    &_raygen_prog_group));
            }

            {
                OptixProgramGroupDesc miss_prog_group_desc = {};
                miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
                miss_prog_group_desc.miss.module = _ptx_miss_module;
                miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
                RENDERTOY3O_OPTIX_CHECK_LOG(optixProgramGroupCreate(
                    _ctx, &miss_prog_group_desc,
                    1, // num program groups
                    &program_group_options,
                    LOG, &LOG_SIZE,
                    &_radiance_miss_group));
            }

            {
                OptixProgramGroupDesc hit_prog_group_desc = {};
                hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                hit_prog_group_desc.hitgroup.moduleCH = _ptx_closehit;
                hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
                RENDERTOY3O_OPTIX_CHECK_LOG(optixProgramGroupCreate(
                    _ctx,
                    &hit_prog_group_desc,
                    1, // num program groups
                    &program_group_options,
                    LOG, &LOG_SIZE,
                    &_radiance_hit_group));
            }

            // TODO: 根据optixProgramGroupCreate的参数可以猜测，传入的OptixProgramGroupDesc可以是一个数组。
            {
                OptixProgramGroupDesc callable_prog_group_desc = {};
                callable_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
                callable_prog_group_desc.callables.moduleDC = _ptx_test;
                callable_prog_group_desc.callables.entryFunctionNameDC = "__direct_callable__test";
                RENDERTOY3O_OPTIX_CHECK_LOG(optixProgramGroupCreate(
                    _ctx,
                    &callable_prog_group_desc,
                    1, // num program groups
                    &program_group_options,
                    LOG, &LOG_SIZE,
                    &_callable_test_group));
            }
        }

        void createPipeline()
        {
            // Optix程序组
            OptixProgramGroup program_groups[] =
                {
                    _raygen_prog_group,
                    _radiance_miss_group,
                    _radiance_hit_group,
                    _callable_test_group,
                };

            // 管线设置中，包含了最大追踪递归深度
            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth = 2;

            RENDERTOY3O_OPTIX_CHECK_LOG(optixPipelineCreate(
                _ctx,
                &_pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof(program_groups) / sizeof(program_groups[0]),
                LOG, &LOG_SIZE,
                &_pipeline));

            // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
            // parameters to optixPipelineSetStackSize.
            // 以下部分都是可选的，用于计算GPU程序栈的大小并且进行程序栈创建。
            // OptixStackSizes stack_sizes = {};
            // RENDERTOY3O_OPTIX_CHECK(optixUtilAccumulateStackSizes(_raygen_prog_group, &stack_sizes, _pipeline));
            // RENDERTOY3O_OPTIX_CHECK(optixUtilAccumulateStackSizes(_radiance_miss_group, &stack_sizes, _pipeline));
            // RENDERTOY3O_OPTIX_CHECK(optixUtilAccumulateStackSizes(_radiance_hit_group, &stack_sizes, _pipeline));
            // RENDERTOY3O_OPTIX_CHECK(optixUtilAccumulateStackSizes(_callable_test_group, &stack_sizes, _pipeline));

            // uint32_t max_trace_depth = 2;
            // uint32_t max_cc_depth = 0;
            // uint32_t max_dc_depth = 0;
            // uint32_t direct_callable_stack_size_from_traversal;
            // uint32_t direct_callable_stack_size_from_state;
            // uint32_t continuation_stack_size;
            // RENDERTOY3O_OPTIX_CHECK(optixUtilComputeStackSizes(
            //     &stack_sizes,
            //     max_trace_depth,
            //     max_cc_depth,
            //     max_dc_depth,
            //     &direct_callable_stack_size_from_traversal,
            //     &direct_callable_stack_size_from_state,
            //     &continuation_stack_size));

            // const uint32_t max_traversal_depth = 2;
            // RENDERTOY3O_OPTIX_CHECK(optixPipelineSetStackSize(
            //     _pipeline,
            //     direct_callable_stack_size_from_traversal,
            //     direct_callable_stack_size_from_state,
            //     continuation_stack_size,
            //     max_traversal_depth));
        }

    public:
        OptixContext()
        {
            RENDERTOY3O_CUDA_CHECK(cudaFree(0));
            CUcontext cu_ctx = 0;
            RENDERTOY3O_OPTIX_CHECK(optixInit());
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &context_log_cb;
            options.logCallbackLevel = 4;
#ifdef DEBUG
            // This may incur significant performance cost and should only be done during development.
            options.validationMode = OPTIX_DEVICE_ctx_VALIDATION_MODE_ALL;
#endif
            RENDERTOY3O_OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &_ctx));

            createModule();
            createProgramGroups();
            createPipeline();
        }
        OptixContext(const OptixContext &) = delete;
        OptixContext(const OptixContext &&) = delete;
        ~OptixContext()
        {
            RENDERTOY3O_OPTIX_CHECK(optixPipelineDestroy(_pipeline));
            RENDERTOY3O_OPTIX_CHECK(optixProgramGroupDestroy(_raygen_prog_group));
            RENDERTOY3O_OPTIX_CHECK(optixProgramGroupDestroy(_radiance_miss_group));
            RENDERTOY3O_OPTIX_CHECK(optixProgramGroupDestroy(_radiance_hit_group));
            RENDERTOY3O_OPTIX_CHECK(optixModuleDestroy(_ptx_module));

            RENDERTOY3O_OPTIX_CHECK(optixDeviceContextDestroy(_ctx));
        }

    public:
        [[nodiscard]] const auto &ctx() const
        {
            return _ctx;
        }
        [[nodiscard]] const auto &raygen_prog_group() const { return _raygen_prog_group; }
        [[nodiscard]] const auto &radiance_miss_group() const { return _radiance_miss_group; }
        [[nodiscard]] const auto &radiance_hit_group() const { return _radiance_hit_group; }
        [[nodiscard]] const auto &callable_test_group() const { return _callable_test_group; }
        [[nodiscard]] const auto &pipeline() const { return _pipeline; }
    };
}