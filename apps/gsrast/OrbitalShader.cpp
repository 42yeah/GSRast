#include "OrbitalShader.hpp"
#include "CameraBase.hpp"
#include "ShaderBase.hpp"
#include "WindowBase.hpp"
#include <glm/gtc/type_ptr.hpp>


OrbitalShader::OrbitalShader(CameraBase::Ptr camera) : ShaderBase("shaders/orbital/vertex.glsl", "shaders/orbital/fragment.glsl")
{
    _camera = camera;
    if (_valid)
    {
        _viewPos = glGetUniformLocation(_program, "view");
        _perspectivePos = glGetUniformLocation(_program, "perspective");
    }
}

void OrbitalShader::use()
{
    glUseProgram(_program);
    glUniformMatrix4fv(_viewPos, 1, GL_FALSE, glm::value_ptr(_camera->getView()));
    glUniformMatrix4fv(_perspectivePos, 1, GL_FALSE, glm::value_ptr(_camera->getPerspective()));
}

bool OrbitalShader::valid()
{
    return _valid;
}
