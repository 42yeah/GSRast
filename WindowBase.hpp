#pragma once

#include <glm/glm.hpp>
#include <memory>
#include "DrawBase.hpp"
#include "Config.hpp"

class WindowBase
{
public:
    CLASS_PTRS(WindowBase)

    virtual bool valid() const = 0;
    virtual bool closed() const = 0;
    virtual float deltaTime() const = 0;
    virtual void pollEvents() = 0;
    virtual void swapBuffers() const = 0;

    virtual void clear(glm::vec4 clearColor) = 0;
    virtual void drawFrame() = 0;
    virtual void addDrawable(std::shared_ptr<DrawBase>) = 0;
    virtual void clearDrawables() = 0;

protected:
    WindowBase();
};
