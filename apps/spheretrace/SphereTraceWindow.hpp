#pragma once

#include "Window.hpp"
#include "Config.hpp"
#include <glm/glm.hpp>


class SphereTraceWindow : public Window
{
public:
    CLASS_PTRS(SphereTraceWindow)

    SphereTraceWindow();
    ~SphereTraceWindow();

protected:
    glm::vec3 _sphereCenter;
};
