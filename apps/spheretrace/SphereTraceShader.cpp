#include "SphereTraceShader.hpp"
#include "CameraBase.hpp"
#include "ShaderBase.hpp"
#include "DrawBase.hpp"
#include <glm/gtc/type_ptr.hpp>


SphereTraceShader::SphereTraceShader(CameraBase::Ptr camera) : ShaderBase("shaders/st/vertex.glsl", "shaders/st/fragment.glsl")
{
    _camera = camera;
    if (_valid)
    {
        _modelPos = glGetUniformLocation(_program, "model");
        _viewPos = glGetUniformLocation(_program, "view");
        _perspectivePos = glGetUniformLocation(_program, "perspective");
        _camPos = glGetUniformLocation(_program, "camPos");
        _sphereCenterPos = glGetUniformLocation(_program, "sphereCenter");
    }
    _sphereCenter = glm::vec3(0.0f);
}

SphereTraceShader::~SphereTraceShader()
{

}

bool SphereTraceShader::valid()
{
    return _valid;
}

void SphereTraceShader::use(const DrawBase &draw)
{
    if (_valid)
    {
        glUseProgram(_program);
        glUniformMatrix4fv(_modelPos, 1, GL_FALSE, glm::value_ptr(draw.getModelMatrix()));
        glUniformMatrix4fv(_viewPos, 1, GL_FALSE, glm::value_ptr(_camera->getView()));
        glUniformMatrix4fv(_perspectivePos, 1, GL_FALSE, glm::value_ptr(_camera->getPerspective()));

        const glm::vec3 &camPos = _camera->getPosition();
        glUniform3f(_camPos, camPos.x, camPos.y, camPos.z);
        glUniform3f(_sphereCenterPos, _sphereCenter.x, _sphereCenter.y, _sphereCenter.z);
    }
}

const glm::vec3 &SphereTraceShader::getCenter() const
{
    return _sphereCenter;
}

void SphereTraceShader::setCenter(const glm::vec3 &center)
{
    _sphereCenter = center;
}
