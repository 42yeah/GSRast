#pragma once

#include "Config.hpp"
#include "DrawBase.hpp"


/**
 * I am a render selector, which on my own, doesn't really
 * do anything. But my insides can be switched up, and therefore
 * achieving the selecting effect.
 */
class RenderSelector : public DrawBase
{
public:
    CLASS_PTRS(RenderSelector)

    RenderSelector();
    ~RenderSelector();

    virtual void draw() override;

    void reset(DrawBase::Ptr drawPtr);

protected:
    DrawBase::Ptr _drawPtr;
};
