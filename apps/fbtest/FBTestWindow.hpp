#pragma once

#include "Config.hpp"
#include "Window.hpp"
#include "Framebuffer.hpp"

class FBTestWindow : public Window
{
public:
    CLASS_PTRS(FBTestWindow)

    FBTestWindow();

    virtual void drawFrame() override;

protected:
    Framebuffer::Ptr _framebuffer;
};
