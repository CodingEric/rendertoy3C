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
#include "cuda/cuda_stream.h"

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
    rendertoy3o::RenderSettings *params = static_cast<rendertoy3o::RenderSettings *>(glfwGetWindowUserPointer(window));
    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
    {
        trackball.setViewMode(sutil::Trackball::LookAtFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->film_settings.width, params->film_settings.height);
        camera_changed = true;
    }
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->film_settings.width, params->film_settings.height);
        camera_changed = true;
    }
}

static void windowSizeCallback(GLFWwindow *window, int32_t res_x, int32_t res_y)
{
    if (minimized)
        return;
    sutil::ensureMinimumSize(res_x, res_y);
    rendertoy3o::RenderSettings *params = static_cast<rendertoy3o::RenderSettings *>(glfwGetWindowUserPointer(window));
    params->film_settings.width = res_x;
    params->film_settings.height = res_y;
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

void initLaunchParams(RenderSettings &params, const CUDAAccel &accel)
{
    RENDERTOY3O_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&params.film_settings.accum_buffer),
        params.film_settings.width * params.film_settings.height * sizeof(float4)));
    params.film_settings.frame_buffer = nullptr; // Will be set when output buffer is mapped

    params.film_settings.samples_per_launch = samples_per_launch;
    params.film_settings.subframe_index = 0u;
    params.handle = accel.ias_handle();
}

void handleCameraUpdate(rendertoy3o::RenderSettings &params)
{
    if (!camera_changed)
        return;
    camera_changed = false;

    camera.setAspectRatio(static_cast<float>(params.film_settings.width) / static_cast<float>(params.film_settings.height));
    params.camera_settings.eye = camera.eye();
    camera.UVWFrame(params.camera_settings.U, params.camera_settings.V, params.camera_settings.W);
}

void handleResize(sutil::CUDAOutputBuffer<uchar4> &output_buffer, rendertoy3o::RenderSettings &params)
{
    if (!resize_dirty)
        return;
    resize_dirty = false;

    output_buffer.resize(params.film_settings.width, params.film_settings.height);

    // Realloc accumulation buffer
    RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(params.film_settings.accum_buffer)));
    RENDERTOY3O_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&params.film_settings.accum_buffer),
        params.film_settings.width * params.film_settings.height * sizeof(float4)));
}

void updateState(sutil::CUDAOutputBuffer<uchar4> &output_buffer, rendertoy3o::RenderSettings &params)
{
    // Update params on device
    if (camera_changed || resize_dirty)
        params.film_settings.subframe_index = 0;

    handleCameraUpdate(params);
    handleResize(output_buffer, params);
}

void launchSubframe(const CUDAStream &stream, sutil::CUDAOutputBuffer<uchar4> &output_buffer, RenderSettings &params, const OptixContext &ctx, const CUDAScene &scene)
{
    // Launch
    uchar4 *result_buffer_data = output_buffer.map();
    params.film_settings.frame_buffer = result_buffer_data;
    scene.update_cuda_params_async(params, stream.stream());

    RENDERTOY3O_OPTIX_CHECK(optixLaunch(
        ctx.pipeline(), 
        stream.stream(),
        scene.params(),
        sizeof(rendertoy3o::RenderSettings),
        &scene.sbt(),
        params.film_settings.width,  // launch width
        params.film_settings.height, // launch height
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
void buildLightSampler(RenderSettings &params)
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
    params.light_settings.light_count = lights.size();
    RENDERTOY3O_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&params.light_settings.lights), lights.size() * sizeof(rendertoy3o::Light)));
    RENDERTOY3O_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(params.light_settings.lights), lights.data(), lights.size() * sizeof(rendertoy3o::Light), cudaMemcpyHostToDevice));
}

void cleanupState(RenderSettings &params)
{
    RENDERTOY3O_CUDA_CHECK(cudaFree(reinterpret_cast<void *>(params.film_settings.accum_buffer)));
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // std::tie(g_meshes, g_textures) = wavefront::loadOBJ({"/home/tianyu/1.obj", "/home/tianyu/2.obj"});
    // std::tie(g_meshes, g_textures) = rendertoy3o::loadOBJ({"/home/tianyu/mat_test.obj"});
    std::tie(g_meshes, g_textures) = rendertoy3o::loadOBJ({"E:\\test.obj"});
    // std::tie(g_meshes, g_textures) = wavefront::loadOBJ({"/run/media/tianyu/hdd0-3d-wksp/testmodels/motion.obj"/*, "/run/media/tianyu/hdd0-3d-wksp/testmodels/motion0002.obj"*/});
    CUDAStream stream;
    OptixContext optix_context;
    CUDAScene cuda_scene(stream, optix_context, g_meshes, g_textures);
    auto params = RenderSettings(768, 768, samples_per_launch, cuda_scene.accel().ias_handle());
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    // 初始化摄像机参数，包括互动场景摄像机
    initCameraState();
    buildLightSampler(params);

    GLFWwindow *window = sutil::initUI("rendertoy3c", params.film_settings.width, params.film_settings.height);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetWindowSizeCallback(window, windowSizeCallback);
    glfwSetWindowIconifyCallback(window, windowIconifyCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetWindowUserPointer(window, &params);

    //
    // Render loop
    //
    {
        sutil::CUDAOutputBuffer<uchar4> output_buffer(
            output_buffer_type,
            params.film_settings.width,
            params.film_settings.height);

        output_buffer.setStream(stream.stream());
        rendertoy3o::GLDisplay gl_display;

        std::chrono::duration<double> state_update_time(0.0);
        std::chrono::duration<double> render_time(0.0);
        std::chrono::duration<double> display_time(0.0);

        do
        {
            auto t0 = std::chrono::steady_clock::now();
            glfwPollEvents();

            updateState(output_buffer, params);
            auto t1 = std::chrono::steady_clock::now();
            state_update_time += t1 - t0;
            t0 = t1;

            launchSubframe(stream, output_buffer, params, optix_context, cuda_scene);
            t1 = std::chrono::steady_clock::now();
            render_time += t1 - t0;
            t0 = t1;

            displaySubframe(output_buffer, gl_display, window);
            t1 = std::chrono::steady_clock::now();
            display_time += t1 - t0;

            sutil::displayStats(state_update_time, render_time, display_time);

            glfwSwapBuffers(window);

            ++params.film_settings.subframe_index;
        } while (!glfwWindowShouldClose(window));
        CUDA_SYNC_CHECK();
    }

    sutil::cleanupUI(window);
    
    cleanupState(params);

    return 0;
}
