#pragma once

#include "Config.hpp"
#include "Window.hpp"
#include "PointCloudShader.hpp"

class GSRastWindow : public Window
{
public:
    CLASS_PTRS(GSRastWindow)

    GSRastWindow();
    ~GSRastWindow();

protected:
    PointCloudShader::Ptr _pcShader;
};
