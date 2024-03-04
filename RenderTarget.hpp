#pragma once

#include "Config.hpp"
#include "DrawBase.hpp"

class RenderTarget
{
public:
    CLASS_PTRS(RenderTarget)

    virtual void clear(glm::vec4 clearColor) = 0;
    virtual void drawFrame() = 0;
    virtual void addDrawable(std::shared_ptr<DrawBase>) = 0;
    virtual void clearDrawables() = 0;
};
