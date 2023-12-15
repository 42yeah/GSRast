#pragma once

#include "Window.hpp"
#include "Config.hpp"
#include "Cube.hpp"
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
    glm::vec3 _sphereCenter;
    Cube::Ptr _cube;
    SphereTraceShader::Ptr _shader;
};
