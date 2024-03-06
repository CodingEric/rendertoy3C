#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>

#include <GLFW/glfw3.h>

#include "shader/shader_data.h"
#include "mesh.h"
#include "light.h"
#include "gui/display.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

bool resize_dirty = false;
bool minimized = false;

// Camera state
bool camera_changed = true;
sutil::Camera camera;
sutil::Trackball trackball;

// Mouse state
int32_t mouse_button = -1;

int32_t samples_per_launch = 8;

//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<wavefront::RayGenData> RayGenRecord;
typedef Record<wavefront::MissData> MissRecord;
typedef Record<wavefront::HitGroupData> HitGroupRecord;
typedef Record<wavefront::CallableData> CallableRecord;

struct Vertex
{
    float x, y, z, pad;
};

struct IndexedTriangle
{
    uint32_t v1, v2, v3, pad;
};

struct Instance
{
    float transform[12];
};

struct PathTracerState
{
    OptixDeviceContext context = 0;

    OptixTraversableHandle gas_handle = 0; // Traversable handle for triangle AS
    CUdeviceptr d_gas_output_buffer = {};  // Triangle AS memory
    std::vector<CUdeviceptr> d_vertices = {};
    std::vector<CUdeviceptr> d_indices = {};
    std::vector<CUdeviceptr> d_normals = {};
    std::vector<CUdeviceptr> d_texcoords = {};

    std::vector<cudaArray_t> textureArrays = {};
    std::vector<cudaTextureObject_t> textureObjects = {};

    OptixModule ptx_module = 0;
    OptixModule ptx_miss_module = 0;
    OptixModule ptx_closehit = 0;
    OptixModule ptx_test = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipeline pipeline = 0;

    OptixProgramGroup raygen_prog_group = 0;
    OptixProgramGroup radiance_miss_group = 0;
    OptixProgramGroup radiance_hit_group = 0;
    OptixProgramGroup callable_test_group = 0;

    CUstream stream = 0;
    wavefront::Params params;
    wavefront::Params *d_params;

    OptixShaderBindingTable sbt = {};
};

//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------

std::vector<wavefront::Mesh> g_meshes;
std::vector<wavefront::Texture> g_textures;

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS)
    {
        mouse_button = button;
        trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
    }
    else
    {
        mouse_button = -1;
    }
}

static void cursorPosCallback(GLFWwindow *window, double xpos, double ypos)
{
    wavefront::Params *params = static_cast<wavefront::Params *>(glfwGetWindowUserPointer(window));

    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
    {
        trackball.setViewMode(sutil::Trackball::LookAtFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
        camera_changed = true;
    }
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
        camera_changed = true;
    }
}

static void windowSizeCallback(GLFWwindow *window, int32_t res_x, int32_t res_y)
{
    // Keep rendering at the current resolution when the window is minimized.
    if (minimized)
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize(res_x, res_y);

    wavefront::Params *params = static_cast<wavefront::Params *>(glfwGetWindowUserPointer(window));
    params->width = res_x;
    params->height = res_y;
    camera_changed = true;
    resize_dirty = true;
}

static void windowIconifyCallback(GLFWwindow *window, int32_t iconified)
{
    minimized = (iconified > 0);
}

static void keyCallback(GLFWwindow *window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
        {
            glfwSetWindowShouldClose(window, true);
        }
    }
    else if (key == GLFW_KEY_G)
    {
        // toggle UI draw
    }
}

static void scrollCallback(GLFWwindow *window, double xscroll, double yscroll)
{
    if (trackball.wheelEvent((int)yscroll))
        camera_changed = true;
}

//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

void printUsageAndExit(const char *argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit(0);
}

void initLaunchParams(PathTracerState &state)
{
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&state.params.accum_buffer),
        state.params.width * state.params.height * sizeof(float4)));
    state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    state.params.samples_per_launch = samples_per_launch;
    state.params.subframe_index = 0u;
    state.params.handle = state.gas_handle;

    CUDA_CHECK(cudaStreamCreate(&state.stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_params), sizeof(wavefront::Params)));
}

void handleCameraUpdate(wavefront::Params &params)
{
    if (!camera_changed)
        return;
    camera_changed = false;

    camera.setAspectRatio(static_cast<float>(params.width) / static_cast<float>(params.height));
    params.eye = camera.eye();
    camera.UVWFrame(params.U, params.V, params.W);
}

void handleResize(sutil::CUDAOutputBuffer<uchar4> &output_buffer, wavefront::Params &params)
{
    if (!resize_dirty)
        return;
    resize_dirty = false;

    output_buffer.resize(params.width, params.height);

    // Realloc accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(params.accum_buffer)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&params.accum_buffer),
        params.width * params.height * sizeof(float4)));
}

void updateState(sutil::CUDAOutputBuffer<uchar4> &output_buffer, wavefront::Params &params)
{
    // Update params on device
    if (camera_changed || resize_dirty)
        params.subframe_index = 0;

    handleCameraUpdate(params);
    handleResize(output_buffer, params);
}

void launchSubframe(sutil::CUDAOutputBuffer<uchar4> &output_buffer, PathTracerState &state)
{
    // Launch
    uchar4 *result_buffer_data = output_buffer.map();
    state.params.frame_buffer = result_buffer_data;
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void *>(state.d_params),
        &state.params, sizeof(wavefront::Params),
        cudaMemcpyHostToDevice, state.stream));

    OPTIX_CHECK(optixLaunch(
        state.pipeline,
        state.stream,
        reinterpret_cast<CUdeviceptr>(state.d_params),
        sizeof(wavefront::Params),
        &state.sbt,
        state.params.width,  // launch width
        state.params.height, // launch height
        1                    // launch depth
        ));
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

void displaySubframe(sutil::CUDAOutputBuffer<uchar4> &output_buffer, wavefront::GLDisplay &gl_display, GLFWwindow *window)
{
    // Display
    int framebuf_res_x = 0; // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0; //
    glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO());
}

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

void initCameraState()
{
    camera.setEye(make_float3(5.0f, 5.0f, 5.0f));
    camera.setLookat(make_float3(0.0f, 1.0f, 0.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFovY(45.0f);
    camera_changed = true;

    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(0.0f, 1.0f, 0.0f));
    trackball.setGimbalLock(true);
}

void createContext(PathTracerState &state)
{
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context;
    CUcontext cu_ctx = 0; // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
#ifdef DEBUG
    // This may incur significant performance cost and should only be done during development.
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

    state.context = context;
}

/// @brief 创建网格加速结构
/// @param state 全程序状态配置
void buildMeshAccel(PathTracerState &state)
{
    // 多mesh的注意事项：
    // 需要使用多组 OptixBuildInput、多组 d_vertices 和多组 d_indices。
    state.d_vertices.resize(g_meshes.size());
    state.d_indices.resize(g_meshes.size());
    state.d_normals.resize(g_meshes.size());
    state.d_texcoords.resize(g_meshes.size());
    std::vector<uint32_t> triangleInputFlags(g_meshes.size());
    std::vector<OptixBuildInput> triangleInputs(g_meshes.size());

    for (size_t i = 0; i < g_meshes.size(); ++i)
    {
        const auto &mesh = g_meshes[i];

        const size_t vertices_size_in_bytes = mesh.vertices.size() * sizeof(float3);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_vertices[i]), vertices_size_in_bytes));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_vertices[i]), mesh.vertices.data(), vertices_size_in_bytes, cudaMemcpyHostToDevice));

        const size_t indices_size_in_bytes = mesh.indices.size() * sizeof(int3);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_indices[i]), indices_size_in_bytes));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_indices[i]), mesh.indices.data(), indices_size_in_bytes, cudaMemcpyHostToDevice));

        const size_t normals_size_in_bytes = mesh.normals.size() * sizeof(float3);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_normals[i]), normals_size_in_bytes));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_normals[i]), mesh.normals.data(), normals_size_in_bytes, cudaMemcpyHostToDevice));

        const size_t texcoords_size_in_bytes = mesh.texcoords.size() * sizeof(float2);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_texcoords[i]), texcoords_size_in_bytes));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_texcoords[i]), mesh.texcoords.data(), texcoords_size_in_bytes, cudaMemcpyHostToDevice));

        auto &triangleInput = triangleInputs[i];
        triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
        triangleInput.triangleArray.numVertices = static_cast<uint32_t>(mesh.vertices.size());
        triangleInput.triangleArray.vertexBuffers = &state.d_vertices[i];

        triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput.triangleArray.indexStrideInBytes = sizeof(int3);
        triangleInput.triangleArray.numIndexTriplets = (int)mesh.indices.size();
        triangleInput.triangleArray.indexBuffer = state.d_indices[i];

        triangleInputFlags[i] = 0;
        triangleInput.triangleArray.flags = &triangleInputFlags[i];
        triangleInput.triangleArray.numSbtRecords = 1;
        triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }

    // 加速结构设置
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION; // 这是加速结构压缩必须的第1个要求
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // 计算加速结构在GPU上[将要]占用的显示内存大小。OptixAccelBufferSizes是一个结构体，内部含有三个参量。这里由于没有更新BVH，所以仅仅采用前两个变量。
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        triangleInputs.data(),
        (int)triangleInputs.size(), // num_build_inputs
        &gas_buffer_sizes));

    // 在设备上申请临时存储空间。
    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output
    // 在设备上申请输出内存空间。
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_buffer_temp_output_gas_and_compacted_size),
        compactedSizeOffset + 8));

    // 配置加速结构构建器以支持大小压缩。
    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE; // 这是加速结构压缩必须的第2个要求
    emitProperty.result = (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    // 执行加速结构构建。
    // 注意在这个构建过程中并没有对BVH进行压缩。
    // 但是这一个过程会计算对BVH压缩以后的大小并且发射到emitProperty中。
    OPTIX_CHECK(optixAccelBuild(
        state.context,
        0, // CUDA stream
        &accel_options,
        triangleInputs.data(),
        (int)triangleInputs.size(), // num build inputs
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &state.gas_handle,
        &emitProperty, // emitted property list
        1              // num emitted properties
        ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer)));
    // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_mat_indices)));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    // 这是技术手册要求的在CPU上执行的判断，因为BVH压缩过程可能在极端情况下导致结果变差。
    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_gas_output_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle));

        CUDA_CHECK(cudaFree((void *)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

/// @brief 使用含有各个pipeline的cu文件创建OptiX模块
/// @param state
void createModule(PathTracerState &state)
{
    OptixPayloadType payloadType = {};
    // radiance prd
    // 辐照度payload，这里是文件结构
    payloadType.numPayloadValues = sizeof(wavefront::radiancePayloadSemantics) / sizeof(wavefront::radiancePayloadSemantics[0]);
    payloadType.payloadSemantics = wavefront::radiancePayloadSemantics;

    OptixModuleCompileOptions module_compile_options = {};
#if !defined(NDEBUG)
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
    module_compile_options.numPayloadTypes = 1;
    module_compile_options.payloadTypes = &payloadType;

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 0;
    state.pipeline_compile_options.numAttributeValues = 2;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    state.pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE; // 根据文档的指示，如果场景中只包含三角形，那么为了最佳性能，应该启用这个flag。

    size_t inputSize = 0;
    // 一个非常神奇的发现：这个程序的.cu文件似乎是动态加载的！（应该是加载了被nvcc编译后的结果）
    const char *input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "raygen.cu", inputSize);

    OPTIX_CHECK_LOG(optixModuleCreate(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input,
        inputSize,
        LOG, &LOG_SIZE,
        &state.ptx_module));

    const char *input_miss = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "miss.cu", inputSize);

    OPTIX_CHECK_LOG(optixModuleCreate(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input_miss,
        inputSize,
        LOG, &LOG_SIZE,
        &state.ptx_miss_module));

    const char *input_closehit = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "closehit_radiance.cu", inputSize);

    OPTIX_CHECK_LOG(optixModuleCreate(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input_closehit,
        inputSize,
        LOG, &LOG_SIZE,
        &state.ptx_closehit));

    const char *input_test = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "test.cu", inputSize);

    OPTIX_CHECK_LOG(optixModuleCreate(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input_test,
        inputSize,
        LOG, &LOG_SIZE,
        &state.ptx_test));
}

/// @brief 创建程序组
/// @param state
void createProgramGroups(PathTracerState &state)
{
    OptixProgramGroupOptions program_group_options = {};

    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = state.ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &raygen_prog_group_desc,
            1, // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.raygen_prog_group));
    }

    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = state.ptx_miss_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &miss_prog_group_desc,
            1, // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.radiance_miss_group));
    }

    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = state.ptx_closehit;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &hit_prog_group_desc,
            1, // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.radiance_hit_group));
    }

    // TODO: 根据optixProgramGroupCreate的参数可以猜测，传入的OptixProgramGroupDesc可以是一个数组。
    {
        OptixProgramGroupDesc callable_prog_group_desc = {};
        callable_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        callable_prog_group_desc.callables.moduleDC = state.ptx_test;
        callable_prog_group_desc.callables.entryFunctionNameDC = "__direct_callable__test";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &callable_prog_group_desc,
            1, // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.callable_test_group));
    }
}

/// @brief 创建流水线
/// @param state
void createPipeline(PathTracerState &state)
{
    // Optix程序组
    OptixProgramGroup program_groups[] =
        {
            state.raygen_prog_group,
            state.radiance_miss_group,
            state.radiance_hit_group,
            state.callable_test_group,
        };

    // 管线设置中，包含了最大追踪递归深度
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;

    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        LOG, &LOG_SIZE,
        &state.pipeline));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    // 以下部分都是可选的，用于计算GPU程序栈的大小并且进行程序栈创建。
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.raygen_prog_group, &stack_sizes, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.radiance_miss_group, &stack_sizes, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.radiance_hit_group, &stack_sizes, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.callable_test_group, &stack_sizes, state.pipeline));

    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size));

    const uint32_t max_traversal_depth = 1;
    OPTIX_CHECK(optixPipelineSetStackSize(
        state.pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth));
}

void createTexture(PathTracerState &state)
{
    int numTextures = g_textures.size();

    state.textureArrays.resize(numTextures);
    state.textureObjects.resize(numTextures);

    for (int i = 0; i < numTextures; ++i)
    {
        auto texture = g_textures[i];

        cudaResourceDesc resDesc = {};

        cudaChannelFormatDesc channel_desc;
        int32_t width = texture.resolution.x;
        int32_t height = texture.resolution.y;
        int32_t numComponents = 4; // 和导入机构是对应的
        int32_t pitch = width * numComponents * sizeof(uint8_t);
        channel_desc = cudaCreateChannelDesc<uchar4>();

        cudaArray_t &pixelArray = state.textureArrays[i];
        CUDA_CHECK(cudaMallocArray(&pixelArray, &channel_desc, width, height));

        CUDA_CHECK(cudaMemcpy2DToArray(pixelArray, 0, 0, texture.pixel.data(), pitch, pitch, height, cudaMemcpyHostToDevice));

        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = pixelArray;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;
        texDesc.maxAnisotropy = 1;
        texDesc.maxMipmapLevelClamp = 99;
        texDesc.minMipmapLevelClamp = 0;
        texDesc.mipmapFilterMode = cudaFilterModePoint;
        texDesc.borderColor[0] = 1.0f;
        texDesc.sRGB = 0;

        cudaTextureObject_t cudaTex = 0;
        CUDA_CHECK(cudaCreateTextureObject(&cudaTex, &resDesc, &texDesc, nullptr));
        state.textureObjects[i] = cudaTex;
    }
}

/// @brief 创建光源采样表
/// @param state
void buildLightSampler(PathTracerState &state)
{
    std::vector<wavefront::Light> lights = {};
    for (const auto &mesh : g_meshes)
    {
        if (length(mesh.material.m_emissive) < 1e-5f)
        {
            continue;
        }

        for (const int3 &triangleIndex : mesh.indices)
        {
            wavefront::Light light = wavefront::Light(mesh.material.m_emissive, mesh.vertices[triangleIndex.x], mesh.vertices[triangleIndex.y], mesh.vertices[triangleIndex.z]);
            lights.push_back(light);
        }
    }

    state.params.light_count = lights.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.params.lights), lights.size() * sizeof(wavefront::Light)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.params.lights), lights.data(), lights.size() * sizeof(wavefront::Light), cudaMemcpyHostToDevice));
}

/// @brief 创建着色器绑定表，和材质强关联
/// @param state
void createSBT(PathTracerState &state)
{
    CUdeviceptr d_raygen_record;
    const size_t raygen_record_size = sizeof(RayGenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_raygen_record), raygen_record_size));

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr d_miss_records;
    const size_t miss_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_miss_records), miss_record_size * wavefront::RAY_TYPE_COUNT));

    MissRecord ms_sbt[1];
    OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_miss_group, &ms_sbt[0]));
    ms_sbt[0].data.bg_color = make_float4(0.0f);

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_miss_records),
        ms_sbt,
        miss_record_size * wavefront::RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice));

    CUdeviceptr d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_hitgroup_records),
        hitgroup_record_size * g_meshes.size()));

    std::vector<HitGroupRecord> hitGroupRecords;
    for (size_t i = 0; i < g_meshes.size(); ++i)
    {
        HitGroupRecord record;
        OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_hit_group, &record));
        // record.data.diffuse_color = {0.8f, 0.8f, 0.8f};
        if (g_meshes[i].material.m_diffuseTextureID != -1)
        {
            record.data.hasTexture = true;
            record.data.texture = state.textureObjects[g_meshes[i].material.m_diffuseTextureID];
        }
        else
        {
            record.data.hasTexture = false;
            record.data.diffuse_color = g_meshes[i].material.m_diffuse;
        }
        record.data.emission_color = g_meshes[i].material.m_emissive;
        record.data.vertices = reinterpret_cast<float3 *>(state.d_vertices[i]);
        record.data.indices = reinterpret_cast<int3 *>(state.d_indices[i]);
        record.data.normals = reinterpret_cast<float3 *>(state.d_normals[i]);
        record.data.texcoords = reinterpret_cast<float2 *>(state.d_texcoords[i]);
        hitGroupRecords.push_back(record);
    }

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_hitgroup_records),
        hitGroupRecords.data(),
        hitgroup_record_size * g_meshes.size(),
        cudaMemcpyHostToDevice));

    CUdeviceptr d_callable_records;
    const size_t callable_record_size = sizeof(CallableRecord);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_callable_records),
        hitgroup_record_size * 1
    ));
    std::vector<CallableRecord> callableRecords;
    for(size_t i = 0; i < 1; ++i)
    {
        CallableRecord record;
        OPTIX_CHECK(optixSbtRecordPackHeader(state.callable_test_group, &record));
        callableRecords.push_back(record);
    }

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_callable_records),
        callableRecords.data(),
        callable_record_size * 1,
        cudaMemcpyHostToDevice
    ));

    state.sbt.raygenRecord = d_raygen_record;
    state.sbt.missRecordBase = d_miss_records;
    state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    state.sbt.missRecordCount = wavefront::RAY_TYPE_COUNT;
    state.sbt.hitgroupRecordBase = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
    state.sbt.hitgroupRecordCount = 1;
    state.sbt.callablesRecordBase = d_callable_records;
    state.sbt.callablesRecordCount = 1;
    state.sbt.callablesRecordStrideInBytes = static_cast<uint32_t>(callable_record_size);
}

void cleanupState(PathTracerState &state)
{
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_miss_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_hit_group));
    OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.hitgroupRecordBase)));
    // TODO: Memory leak.
    // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_vertices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_gas_output_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_params)));
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // 总体状态机
    PathTracerState state;
    state.params.width = 768;
    state.params.height = 768;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    //
    // Parse command line options
    //
    std::string outfile;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "--no-gl-interop")
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if (arg == "--file" || arg == "-f")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            outfile = argv[++i];
        }
        else if (arg.substr(0, 6) == "--dim=")
        {
            const std::string dims_arg = arg.substr(6);
            int w, h;
            sutil::parseDimensions(dims_arg.c_str(), w, h);
            state.params.width = w;
            state.params.height = h;
        }
        else if (arg == "--launch-samples" || arg == "-s")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            samples_per_launch = atoi(argv[++i]);
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    try
    {
        // 初始化摄像机参数，包括互动场景摄像机
        initCameraState();

        //
        // Set up OptiX state
        //
        std::tie(g_meshes, g_textures) = wavefront::loadOBJ("/home/tianyu/1.obj");

        // 创建CUDA和OptiX上下文
        createContext(state);
        // 创建网格加速结构
        buildMeshAccel(state);
        // 创建模块
        createModule(state);
        // 创建程序组
        createProgramGroups(state);
        createPipeline(state);
        // 创建贴图
        createTexture(state);
        // 创建光源列表
        buildLightSampler(state);
        // 创建着色器绑定表
        createSBT(state);

        initLaunchParams(state);

        if (outfile.empty())
        {
            GLFWwindow *window = sutil::initUI("Project Wavefront", state.params.width, state.params.height);
            glfwSetMouseButtonCallback(window, mouseButtonCallback);
            glfwSetCursorPosCallback(window, cursorPosCallback);
            glfwSetWindowSizeCallback(window, windowSizeCallback);
            glfwSetWindowIconifyCallback(window, windowIconifyCallback);
            glfwSetKeyCallback(window, keyCallback);
            glfwSetScrollCallback(window, scrollCallback);
            glfwSetWindowUserPointer(window, &state.params);

            //
            // Render loop
            //
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height);

                output_buffer.setStream(state.stream);
                wavefront::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time(0.0);
                std::chrono::duration<double> render_time(0.0);
                std::chrono::duration<double> display_time(0.0);

                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState(output_buffer, state.params);
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe(output_buffer, state);
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe(output_buffer, gl_display, window);
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats(state_update_time, render_time, display_time);

                    glfwSwapBuffers(window);

                    ++state.params.subframe_index;
                } while (!glfwWindowShouldClose(window));
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI(window);
        }
        else
        {
            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                sutil::initGLFW(); // For GL context
                sutil::initGL();
            }

            {
                // this scope is for output_buffer, to ensure the destructor is called bfore glfwTerminate()

                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height);

                handleCameraUpdate(state.params);
                handleResize(output_buffer, state.params);
                launchSubframe(output_buffer, state);

                sutil::ImageBuffer buffer;
                buffer.data = output_buffer.getHostPointer();
                buffer.width = output_buffer.width();
                buffer.height = output_buffer.height();
                buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

                sutil::saveImage(outfile.c_str(), buffer, false);
            }

            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                glfwTerminate();
            }
        }

        cleanupState(state);
    }
    catch (std::exception &e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
