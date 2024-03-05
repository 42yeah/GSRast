#include "Framebuffer.hpp"
#include "Config.hpp"
#include "DrawBase.hpp"
#include "RenderTarget.hpp"

float Framebuffer::rectData[18] = {
    -1.0f, -1.0f, 0.0f,
    1.0f, -1.0f, 0.0f,
    1.0f, 1.0f, 0.0f,
    1.0f, 1.0f, 0.0f,
    -1.0f, 1.0f, 0.0f,
    -1.0f, -1.0f, 0.0f
};


Framebuffer::Framebuffer(int width, int height, bool hdr, ShaderBase::Ptr shader) : RenderTarget(), BufferGeo()
{
    _width = width;
    _height = height;
    _shader = shader;

    glGenTextures(1, &_texture);
    glBindTexture(GL_TEXTURE_2D, _texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    if (!hdr)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    }
    else
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    }

    glGenRenderbuffers(1, &_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, _rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);

    glGenFramebuffers(1, &_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _texture, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, _rbo);

    glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);

    _hdr = hdr;
    _clearColor = glm::vec4(1.0, 0.0, 1.0, 1.0);

    // Configure the geometry (basically a rect)
    configure(rectData, 6, sizeof(rectData), shader);
}

Framebuffer::~Framebuffer()
{
    glDeleteFramebuffers(1, &_fbo);
    glDeleteBuffers(1, &_rbo);
    glDeleteTextures(1, &_texture);
}

void Framebuffer::clear(glm::vec4 clearColor)
{
    _clearColor = clearColor;
}

void Framebuffer::drawFrame()
{
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
    stacks().viewportStack.push({ 0, 0, _width, _height });
    stacks().framebufferStack.push(_fbo);
    glViewport(0, 0, _width, _height);
    glClearColor(_clearColor.x, _clearColor.y, _clearColor.z, _clearColor.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for (const DrawBase::Ptr &draw : _drawList)
    {
        draw->draw();
    }
    stacks().viewportStack.pop();
    stacks().framebufferStack.pop();

    const Viewport &oldStack = stacks().viewportStack.top();
    glViewport(oldStack.x, oldStack.y, oldStack.w, oldStack.h);
    if (!stacks().framebufferStack.empty())
    {
        glBindFramebuffer(GL_FRAMEBUFFER, stacks().framebufferStack.top());
    }
    else
    {
        glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE);
    }
}

void Framebuffer::addDrawable(DrawBase::Ptr draw)
{
    _drawList.push_back(draw);
}

void Framebuffer::clearDrawables()
{
    _drawList.clear();
}

void Framebuffer::draw()
{
    BufferGeo::draw();
}

GLuint Framebuffer::getTexture() const
{
    return _texture;
}

int Framebuffer::getWidth() const
{
    return _width;
}

int Framebuffer::getHeight() const
{
    return _height;
}

