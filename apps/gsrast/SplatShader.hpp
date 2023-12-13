#pragma once

#include "DrawBase.hpp"
#include "ShaderBase.hpp"
#include "Config.hpp"
#include "CameraBase.hpp"
#include "apps/gsrast/GSEllipsoids.hpp"

// The GSS shader is actually not different to the GSPC shader,
// and their only difference is their source.
class SplatShader : public ShaderBase
{
public:
    CLASS_PTRS(SplatShader)

    SplatShader(CameraBase::Ptr camera);

    virtual void use(const DrawBase &draw) override;
    virtual void use(const GSEllipsoids &ellipsoids);
    virtual bool valid() override;

protected:
    CameraBase::Ptr _camera;
    GLuint _modelPos, _viewPos, _perspectivePos;

};
