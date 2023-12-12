#pragma once

#include "Config.hpp"
#include "Window.hpp"
#include "OrbitalShader.hpp"

class GSRastWindow : public Window
{
public:
    CLASS_PTRS(GSRastWindow)

    GSRastWindow();
    ~GSRastWindow();

protected:
    OrbitalShader::Ptr _orbitalShader;
};
