#pragma once

#include "ShaderBase.hpp"
#include "Config.hpp"
#include "CameraBase.hpp"


class OrbitalShader : public ShaderBase
{
public:
    CLASS_PTRS(OrbitalShader)

    OrbitalShader(CameraBase::Ptr camera);

    virtual void use() override;
    virtual bool valid() override;

protected:
    CameraBase::Ptr _camera;
    GLuint _perspectivePos;
    GLuint _viewPos;
};
