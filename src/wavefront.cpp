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
#include "cuda/cuda_scene.h"

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

using namespace rendertoy3o;

struct PathTracerState
{
    CUstream stream = 0;
    rendertoy3o::Params params;
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

void initLaunchParams(PathTracerState &state, const CUDAAccel &accel)
{
    RENDERTOY3O_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&state.params.accum_buffer),
        state.params.width * state.params.height * sizeof(float4)));
    state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    state.params.samples_per_launch = samples_per_launch;
    state.params.subframe_index = 0u;
    state.params.handle = accel.ias_handle();

    RENDERTOY3O_CUDA_CHECK(cudaStreamCreate(&state.stream));
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

void launchSubframe(sutil::CUDAOutputBuffer<uchar4> &output_buffer, PathTracerState &state, const OptixContext &ctx, const CUDAScene &scene)
{
    // Launch
    uchar4 *result_buffer_data = output_buffer.map();
    state.params.frame_buffer = result_buffer_data;
    scene.update_cuda_params_async(state.params, state.stream);

    RENDERTOY3O_OPTIX_CHECK(optixLaunch(
        ctx.pipeline(),
        state.stream,
        scene.params(),
        sizeof(rendertoy3o::Params),
        &scene.sbt(),
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

void cleanupState(PathTracerState &state)
{
    RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.params.accum_buffer)));
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

    // 初始化摄像机参数，包括互动场景摄像机
    initCameraState();

    // std::tie(g_meshes, g_textures) = wavefront::loadOBJ({"/home/tianyu/1.obj", "/home/tianyu/2.obj"});
    std::tie(g_meshes, g_textures) = rendertoy3o::loadOBJ({"/home/tianyu/mat_test.obj"});
    // std::tie(g_meshes, g_textures) = wavefront::loadOBJ({"/run/media/tianyu/hdd0-3d-wksp/testmodels/motion.obj"/*, "/run/media/tianyu/hdd0-3d-wksp/testmodels/motion0002.obj"*/});

    OptixContext optix_context;
    CUDAScene cuda_scene(optix_context, g_meshes, g_textures);

    // 创建光源列表
    buildLightSampler(state);

    initLaunchParams(state, cuda_scene.accel());

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

            launchSubframe(output_buffer, state, optix_context, cuda_scene);
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

    cleanupState(state);

    return 0;
}
