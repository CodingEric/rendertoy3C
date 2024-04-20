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
#include "util/exception.h"
#include "cuda/cuda_buffer.h"
#include "cuda/cuda_texture.h"
#include "cuda/cuda_mesh.h"
#include "cuda/cuda_accel.h"
#include "cuda/optix_context.h"

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

typedef Record<rendertoy3o::RayGenData> RayGenRecord;
typedef Record<rendertoy3o::MissData> MissRecord;
typedef Record<rendertoy3o::HitGroupData> HitGroupRecord;
typedef Record<rendertoy3o::CallableData> CallableRecord;

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

using namespace rendertoy3o;

struct PathTracerState
{
    std::vector<OptixTraversableHandle> gas_motion_handle = {};

    std::vector<CUDAMesh> d_meshes = {};

    std::vector<std::shared_ptr<CUDATexture<uchar4>>> textures = {};

    std::vector<CUdeviceptr> motion_transform = {};

    OptixContext optix_context;

    CUstream stream = 0;
    rendertoy3o::Params params;
    rendertoy3o::Params *d_params;

    OptixShaderBindingTable sbt = {};

    CUDAAccel accel;
};

//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------

std::vector<rendertoy3o::Mesh> g_meshes;
std::vector<rendertoy3o::Texture> g_textures;

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
    rendertoy3o::Params *params = static_cast<rendertoy3o::Params *>(glfwGetWindowUserPointer(window));

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

    rendertoy3o::Params *params = static_cast<rendertoy3o::Params *>(glfwGetWindowUserPointer(window));
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
    RENDERTOY3O_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&state.params.accum_buffer),
        state.params.width * state.params.height * sizeof(float4)));
    state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    state.params.samples_per_launch = samples_per_launch;
    state.params.subframe_index = 0u;
    state.params.handle = state.accel.ias_handle();

    RENDERTOY3O_CUDA_CHECK(cudaStreamCreate(&state.stream));
    RENDERTOY3O_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_params), sizeof(rendertoy3o::Params)));
}

void handleCameraUpdate(rendertoy3o::Params &params)
{
    if (!camera_changed)
        return;
    camera_changed = false;

    camera.setAspectRatio(static_cast<float>(params.width) / static_cast<float>(params.height));
    params.eye = camera.eye();
    camera.UVWFrame(params.U, params.V, params.W);
}

void handleResize(sutil::CUDAOutputBuffer<uchar4> &output_buffer, rendertoy3o::Params &params)
{
    if (!resize_dirty)
        return;
    resize_dirty = false;

    output_buffer.resize(params.width, params.height);

    // Realloc accumulation buffer
    RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(params.accum_buffer)));
    RENDERTOY3O_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&params.accum_buffer),
        params.width * params.height * sizeof(float4)));
}

void updateState(sutil::CUDAOutputBuffer<uchar4> &output_buffer, rendertoy3o::Params &params)
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
    RENDERTOY3O_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void *>(state.d_params),
        &state.params, sizeof(rendertoy3o::Params),
        cudaMemcpyHostToDevice, state.stream));

    RENDERTOY3O_OPTIX_CHECK(optixLaunch(
        state.optix_context.pipeline(),
        state.stream,
        reinterpret_cast<CUdeviceptr>(state.d_params),
        sizeof(rendertoy3o::Params),
        &state.sbt,
        state.params.width,  // launch width
        state.params.height, // launch height
        1                    // launch depth
        ));
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

void displaySubframe(sutil::CUDAOutputBuffer<uchar4> &output_buffer, rendertoy3o::GLDisplay &gl_display, GLFWwindow *window)
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

/// @brief 创建网格加速结构
/// @param state 全程序状态配置
void buildMeshAccel(PathTracerState &state)
{
    // 多mesh的注意事项：
    // 需要使用多组 OptixBuildInput、多组 d_vertices 和多组 d_indices。
    state.motion_transform.resize(g_meshes.size());
    state.gas_motion_handle.resize(g_meshes.size());

    for (size_t i = 0; i < g_meshes.size(); ++i)
    {
        const auto &mesh = g_meshes[i];
        state.d_meshes.push_back(CUDAMesh(state.optix_context.ctx(), mesh));

        // {
        //     const float motion_matrix_keys[2][12] =
        //         {
        //             {1.0f, 0.0f, 0.0f, 0.0f,
        //              0.0f, 1.0f, 0.0f, 0.0f,
        //              0.0f, 0.0f, 1.0f, 0.0f},
        //             {1.0f, 0.0f, 0.0f, 0.0f,
        //              0.0f, 1.0f, 0.0f, 0.5f,
        //              0.0f, 0.0f, 1.0f, 0.0f}};

        //     OptixMatrixMotionTransform motion_transform = {};
        //     motion_transform.child = state.gas_handle[i]; // 这里需要拿到 GAS 的handle。
        //     motion_transform.motionOptions.numKeys = 2;
        //     motion_transform.motionOptions.timeBegin = 0.0f;
        //     motion_transform.motionOptions.timeEnd = 1.0f;
        //     motion_transform.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
        //     memcpy(motion_transform.transform, motion_matrix_keys, 2 * 12 * sizeof(float));

        //     RENDERTOY3O_CUDA_CHECK(cudaMalloc(
        //         reinterpret_cast<void **>(&state.motion_transform[i]),
        //         sizeof(OptixMatrixMotionTransform)));

        //     RENDERTOY3O_CUDA_CHECK(cudaMemcpy(
        //         reinterpret_cast<void *>(state.motion_transform[i]),
        //         &motion_transform,
        //         sizeof(OptixMatrixMotionTransform),
        //         cudaMemcpyHostToDevice));

        //     RENDERTOY3O_OPTIX_CHECK(optixConvertPointerToTraversableHandle(
        //         state.optix_context.ctx(),
        //         state.motion_transform[i],
        //         OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM,
        //         &state.gas_motion_handle[i])); // 运动模糊：这说明了几何级别变形会生成一个独立于原有gas_handle的新motion_gas_handle。这个handle可以被插入全局handle，也可以被用于TLAS。

        //     // RENDERTOY3O_CUDA_CHECK(cudaFree((void *)state.motion_transform[i]));
        // }
    }
}

void buildInstanceAccel(PathTracerState &state)
{
    float transformation[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    for(const auto &mesh : state.d_meshes)
    {
        state.accel.append_instance(mesh, transformation);
    }
    state.accel.build(state.optix_context.ctx());
}

void createTexture(PathTracerState &state)
{
    int numTextures = g_textures.size();
    state.textures.resize(numTextures);
    for (int i = 0; i < numTextures; ++i)
    {
        auto texture = g_textures[i];
        state.textures[i] = std::make_shared<CUDATexture<uchar4>>(texture.resolution.x,
                                                                  texture.resolution.y,
                                                                  texture.pixel.data(),
                                                                  CUDATexture<uchar4>::AddressMode::Wrap,
                                                                  CUDATexture<uchar4>::FilterMode::Linear);
    }
}

/// @brief 创建光源采样表
/// @param state
void buildLightSampler(PathTracerState &state)
{
    std::vector<rendertoy3o::Light> lights = {};
    for (const auto &mesh : g_meshes)
    {
        if (length(mesh.material.m_emissive) < 1e-5f)
        {
            continue;
        }
        for (const int3 &triangleIndex : mesh.indices)
        {
            rendertoy3o::Light light = rendertoy3o::Light(mesh.material.m_emissive, mesh.vertices[0][triangleIndex.x], mesh.vertices[0][triangleIndex.y], mesh.vertices[0][triangleIndex.z]);
            lights.push_back(light);
        }
    }
    state.params.light_count = lights.size();
    RENDERTOY3O_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.params.lights), lights.size() * sizeof(rendertoy3o::Light)));
    RENDERTOY3O_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.params.lights), lights.data(), lights.size() * sizeof(rendertoy3o::Light), cudaMemcpyHostToDevice));
}

/// @brief 创建着色器绑定表，和材质强关联
/// @param state
void createSBT(PathTracerState &state)
{
    CUdeviceptr d_raygen_record;
    const size_t raygen_record_size = sizeof(RayGenRecord);
    RENDERTOY3O_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_raygen_record), raygen_record_size));

    RayGenRecord rg_sbt = {};
    RENDERTOY3O_OPTIX_CHECK(optixSbtRecordPackHeader(state.optix_context.raygen_prog_group(), &rg_sbt));

    RENDERTOY3O_CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr d_miss_records;
    const size_t miss_record_size = sizeof(MissRecord);
    RENDERTOY3O_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_miss_records), miss_record_size * rendertoy3o::RAY_TYPE_COUNT));

    MissRecord ms_sbt[1];
    RENDERTOY3O_OPTIX_CHECK(optixSbtRecordPackHeader(state.optix_context.radiance_miss_group(), &ms_sbt[0]));
    ms_sbt[0].data.bg_color = make_float4(0.0f);

    RENDERTOY3O_CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_miss_records),
        ms_sbt,
        miss_record_size * rendertoy3o::RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice));

    CUdeviceptr d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    RENDERTOY3O_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_hitgroup_records),
        hitgroup_record_size * g_meshes.size()));

    std::vector<HitGroupRecord> hitGroupRecords;
    for (size_t i = 0; i < g_meshes.size(); ++i)
    {
        HitGroupRecord record;
        RENDERTOY3O_OPTIX_CHECK(optixSbtRecordPackHeader(state.optix_context.radiance_hit_group(), &record));
        // record.data.diffuse_color = {0.8f, 0.8f, 0.8f};
        if (g_meshes[i].material.m_diffuseTextureID != -1)
        {
            record.data.hasTexture = true;
            record.data.texture = state.textures[g_meshes[i].material.m_diffuseTextureID]->texture_object();
        }
        else
        {
            record.data.hasTexture = false;
            record.data.diffuse_color = g_meshes[i].material.m_diffuse;
        }
        record.data.emission_color = g_meshes[i].material.m_emissive;
        record.data.vertices = reinterpret_cast<float3 *>(state.d_meshes[i].vertex_buffer().buffer_ptr());
        record.data.indices = reinterpret_cast<int3 *>(state.d_meshes[i].index_buffer().buffer_ptr());
        record.data.normals = reinterpret_cast<float3 *>(state.d_meshes[i].normal_buffer().buffer_ptr());
        record.data.texcoords = reinterpret_cast<float2 *>(state.d_meshes[i].texcoord_buffer().buffer_ptr());
        hitGroupRecords.push_back(record);
    }

    RENDERTOY3O_CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_hitgroup_records),
        hitGroupRecords.data(),
        hitgroup_record_size * g_meshes.size(),
        cudaMemcpyHostToDevice));

    CUdeviceptr d_callable_records;
    const size_t callable_record_size = sizeof(CallableRecord);
    RENDERTOY3O_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_callable_records),
        callable_record_size * 1));
    std::vector<CallableRecord> callableRecords;
    for (size_t i = 0; i < 1; ++i)
    {
        CallableRecord record;
        RENDERTOY3O_OPTIX_CHECK(optixSbtRecordPackHeader(state.optix_context.callable_test_group(), &record));
        callableRecords.push_back(record);
    }

    RENDERTOY3O_CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_callable_records),
        callableRecords.data(),
        callable_record_size * 1,
        cudaMemcpyHostToDevice));

    state.sbt.raygenRecord = d_raygen_record;
    state.sbt.missRecordBase = d_miss_records;
    state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    state.sbt.missRecordCount = rendertoy3o::RAY_TYPE_COUNT;
    state.sbt.hitgroupRecordBase = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
    state.sbt.hitgroupRecordCount = g_meshes.size();
    state.sbt.callablesRecordBase = d_callable_records;
    state.sbt.callablesRecordCount = 1;
    state.sbt.callablesRecordStrideInBytes = static_cast<uint32_t>(callable_record_size);
}

void cleanupState(PathTracerState &state)
{
    RENDERTOY3O_OPTIX_CHECK(optixDeviceContextDestroy(state.optix_context.ctx()));
    RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
    RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.missRecordBase)));
    RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.hitgroupRecordBase)));
    RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.params.accum_buffer)));
    RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_params)));
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

    // try
    // {
    // 初始化摄像机参数，包括互动场景摄像机
    initCameraState();

    //
    // Set up OptiX state
    //
    // std::tie(g_meshes, g_textures) = wavefront::loadOBJ({"/home/tianyu/1.obj", "/home/tianyu/2.obj"});
    std::tie(g_meshes, g_textures) = rendertoy3o::loadOBJ({"/home/tianyu/mat_test.obj"});
    // std::tie(g_meshes, g_textures) = wavefront::loadOBJ({"/run/media/tianyu/hdd0-3d-wksp/testmodels/motion.obj"/*, "/run/media/tianyu/hdd0-3d-wksp/testmodels/motion0002.obj"*/});

    // 创建网格加速结构
    buildMeshAccel(state);
    // 创建层次化实例加速结构
    buildInstanceAccel(state);
    // 创建贴图
    createTexture(state);
    // 创建光源列表
    buildLightSampler(state);
    // 创建着色器绑定表
    createSBT(state);

    initLaunchParams(state);

    // if (outfile.empty())
    // {
        GLFWwindow *window = sutil::initUI("rendertoy3c", state.params.width, state.params.height);
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
            rendertoy3o::GLDisplay gl_display;

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
    // }
    // else
    // {
    //     if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
    //     {
    //         sutil::initGLFW(); // For GL context
    //         sutil::initGL();
    //     }

    //     {
    //         // this scope is for output_buffer, to ensure the destructor is called bfore glfwTerminate()

    //         sutil::CUDAOutputBuffer<uchar4> output_buffer(
    //             output_buffer_type,
    //             state.params.width,
    //             state.params.height);

    //         handleCameraUpdate(state.params);
    //         handleResize(output_buffer, state.params);
    //         launchSubframe(output_buffer, state);

    //         sutil::ImageBuffer buffer;
    //         buffer.data = output_buffer.getHostPointer();
    //         buffer.width = output_buffer.width();
    //         buffer.height = output_buffer.height();
    //         buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

    //         sutil::saveImage(outfile.c_str(), buffer, false);
    //     }

    //     if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
    //     {
    //         glfwTerminate();
    //     }
    // }

    cleanupState(state);
    // }
    // catch (std::exception &e)
    // {
    //     std::cerr << "Caught exception: " << e.what() << "\n";
    //     return 1;
    // }

    return 0;
}
