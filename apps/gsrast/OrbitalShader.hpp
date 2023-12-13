#pragma once

#include "ShaderBase.hpp"
#include "Config.hpp"
#include "CameraBase.hpp"


class OrbitalShader : public ShaderBase
{
public:
    CLASS_PTRS(OrbitalShader)

    OrbitalShader(CameraBase::Ptr camera);

    virtual void use(const DrawBase &draw) override;
    virtual bool valid() override;

protected:
    CameraBase::Ptr _camera;
    GLuint _modelPos;
    GLuint _perspectivePos;
    GLuint _viewPos;
};
