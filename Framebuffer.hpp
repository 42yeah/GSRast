#pragma once

#include <vector>
#include <glad/glad.h>
#include "Config.hpp"
#include "RenderTarget.hpp"
#include "DrawBase.hpp"
#include "BufferGeo.hpp"

/**
 * Framebuffer is a drawable render target. That means:
 * Framebuffers can be drawn onto;
 * And framebuffers can be drawn onto other stuffs.
 */
class Framebuffer : public RenderTarget, public BufferGeo
{
public:
    CLASS_PTRS(Framebuffer)

    Framebuffer(int width, int height, bool hdr, ShaderBase::Ptr shader);
    ~Framebuffer();

    virtual void clear(glm::vec4 clearColor) override;
    virtual void drawFrame() override;
    virtual void addDrawable(DrawBase::Ptr draw) override;
    virtual void clearDrawables() override;

    virtual void draw() override;

    GLuint getTexture() const;

    static float rectData[18];

protected:
    int _width, _height; // Framebuffer dimension
    ShaderBase::Ptr _shader; // shader used to draw this framebuffer
    std::vector<DrawBase::Ptr> _drawList;
    GLuint _fbo, _rbo, _texture;
    bool _hdr;
    glm::vec4 _clearColor;
};
