#pragma once

#include <glm/glm.hpp>
#include "Config.hpp"

class WindowBase;

class CameraBase
{
public:
    CLASS_PTRS(CameraBase)

    virtual const glm::mat4 &getView() const = 0;
    virtual const glm::mat4 &getPerspective() const = 0;
    virtual void update(const WindowBase &window) = 0;

protected:
    CameraBase();
};
