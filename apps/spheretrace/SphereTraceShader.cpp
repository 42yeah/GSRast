#include "SphereTraceShader.hpp"
#include "CameraBase.hpp"
#include "ShaderBase.hpp"
#include "DrawBase.hpp"
#include "apps/spheretrace/Ellipsoid.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <memory>


SphereTraceShader::SphereTraceShader(CameraBase::Ptr camera) : ShaderBase("shaders/st/vertex.glsl", "shaders/st/fragment.glsl")
{
    _camera = camera;
    _cubeMode = false;
    if (_valid)
    {
        _modelPos = glGetUniformLocation(_program, "model");
        _viewPos = glGetUniformLocation(_program, "view");
        _perspectivePos = glGetUniformLocation(_program, "perspective");
        _camPos = glGetUniformLocation(_program, "camPos");
        _sphereCenterPos = glGetUniformLocation(_program, "sphereCenter");
        _sphereScalePos = glGetUniformLocation(_program, "sphereScale");
        _cubeModePos = glGetUniformLocation(_program, "cubeMode");
        _sphereRotationPos = glGetUniformLocation(_program, "sphereRotation");
    }
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

        // We have to be actually rendering an ellipsoid
        const Ellipsoid &ellipsoid = dynamic_cast<const Ellipsoid &>(draw);
        const glm::vec3 &center = ellipsoid.getCenter();
        const glm::vec3 &scale = ellipsoid.getScale();
        glUniform3f(_sphereCenterPos, center.x, center.y, center.z);
        glUniform3f(_sphereScalePos, scale.x, scale.y, scale.z);
        glUniform1i(_cubeModePos, _cubeMode);
        glUniformMatrix3fv(_sphereRotationPos, 1, GL_FALSE, glm::value_ptr(ellipsoid.getRotationMatrix()));
    }
}

bool SphereTraceShader::toggleCubeMode()
{
    _cubeMode = !_cubeMode;
    return _cubeMode;
}
