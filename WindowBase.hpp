#pragma once

#include <glm/glm.hpp>
#include <memory>
#include "DrawBase.hpp"
#include "Config.hpp"
#include "RenderTarget.hpp"
#include "CameraBase.hpp"

class CameraBase;

class WindowBase : public RenderTarget
{
public:
    CLASS_PTRS(WindowBase)

    virtual bool valid() const = 0;
    virtual bool closed() const = 0;
    virtual float deltaTime() const = 0;
    virtual void pollEvents() = 0;
    virtual void swapBuffers() const = 0;

    // WindowBase does not implemente RenderTarget functions
    virtual void clear(glm::vec4 clearColor) = 0;
    virtual void drawFrame() = 0;
    virtual void addDrawable(std::shared_ptr<DrawBase>) = 0;
    virtual void clearDrawables() = 0;

    virtual int getWidth() const = 0;
    virtual int getHeight() const = 0;
    CameraBase::Ptr getCamera() const;
    void setCamera(CameraBase::Ptr camera);

protected:
    WindowBase();

    CameraBase::Ptr _camera;
};
