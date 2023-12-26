#pragma once

#include "Config.hpp"
#include "Window.hpp"
#include "PointCloudShader.hpp"
#include "SplatShader.hpp"
#include "CopyShader.hpp"

class GSRastWindow : public Window
{
public:
    CLASS_PTRS(GSRastWindow)

    GSRastWindow();
    ~GSRastWindow();

    virtual void keyCallback(int key, int scancode, int action, int mods) override;

protected:
    PointCloudShader::Ptr _pcShader;
    SplatShader::Ptr _splatShader;
    CopyShader::Ptr _copyShader;
};
