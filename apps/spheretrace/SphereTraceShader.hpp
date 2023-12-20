#pragma once

#include "ShaderBase.hpp"
#include "CameraBase.hpp"
#include "Config.hpp"
#include <glm/glm.hpp>


class SphereTraceShader : public ShaderBase
{
public:
    CLASS_PTRS(SphereTraceShader)

    SphereTraceShader(CameraBase::Ptr camera);
    ~SphereTraceShader();

    virtual bool valid() override;
    virtual void use(const DrawBase &draw) override;

    bool toggleCubeMode();

protected:
    bool _cubeMode;

    GLuint _modelPos, _viewPos, _perspectivePos;
    GLuint _camPos, _sphereCenterPos, _sphereScalePos, _cubeModePos, _sphereRotationPos;
    CameraBase::Ptr _camera;
};
