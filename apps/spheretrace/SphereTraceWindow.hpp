#pragma once

#include "Window.hpp"
#include "Config.hpp"
#include "Ellipsoid.hpp"
#include "SphereTraceShader.hpp"
#include <glm/glm.hpp>


class SphereTraceWindow : public Window
{
public:
    CLASS_PTRS(SphereTraceWindow)

    SphereTraceWindow();
    ~SphereTraceWindow();

    virtual void pollEvents() override;

protected:
    virtual void keyCallback(int key, int scancode, int action, int mods) override;

    Ellipsoid::Ptr _ellipsoid;
    SphereTraceShader::Ptr _shader;
    int _currentAxis;
};
