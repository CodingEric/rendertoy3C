#pragma once

#include <glad/glad.h>

#include <cstdint>
#include <string>

#include <sutil/sutil.h>
#include <sutil/sutilapi.h>

namespace rendertoy3o
{

class GLDisplay
{
public:
    GLDisplay(
        sutil::BufferImageFormat format = sutil::BufferImageFormat::UNSIGNED_BYTE4);

    void display(
            const int32_t  screen_res_x,
            const int32_t  screen_res_y,
            const int32_t  framebuf_res_x,
            const int32_t  framebuf_res_y,
            const uint32_t pbo) const;

private:
    GLuint   m_render_tex = 0u;
    GLuint   m_program = 0u;
    GLint    m_render_tex_uniform_loc = -1;
    GLuint   m_quad_vertex_buffer = 0;

    sutil::BufferImageFormat m_image_format;

    static const std::string s_vert_source;
    static const std::string s_frag_source;
};

} // namespace rendertoy3o
