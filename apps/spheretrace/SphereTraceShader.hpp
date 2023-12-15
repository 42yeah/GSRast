#pragma once

#include "ShaderBase.hpp"
#include "CameraBase.hpp"
#include "Config.hpp"


class SphereTraceShader : public ShaderBase
{
public:
    CLASS_PTRS(SphereTraceShader)

    SphereTraceShader(CameraBase::Ptr camera);
    ~SphereTraceShader();

    virtual bool valid() override;
    virtual void use(const DrawBase &draw) override;

protected:
    GLuint _modelPos, _viewPos, _perspectivePos;
    GLuint _camPos;
    CameraBase::Ptr _camera;
};
