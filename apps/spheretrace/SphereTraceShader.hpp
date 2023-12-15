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

    const glm::vec3 &getCenter() const;
    void setCenter(const glm::vec3 &center);

protected:
    GLuint _modelPos, _viewPos, _perspectivePos;
    GLuint _camPos, _sphereCenterPos;
    CameraBase::Ptr _camera;
    glm::vec3 _sphereCenter;
};
