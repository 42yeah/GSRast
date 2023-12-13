#include "SplatShader.hpp"
#include "ShaderBase.hpp"
#include "DrawBase.hpp"
#include <glm/gtc/type_ptr.hpp>


SplatShader::SplatShader(CameraBase::Ptr camera) : ShaderBase("shaders/splats/vertex.glsl", "shaders/splats/fragment.glsl")
{
    _camera = camera;
    if (_valid)
    {
        _modelPos = glGetUniformLocation(_program, "model");
        _viewPos = glGetUniformLocation(_program, "view");
        _perspectivePos = glGetUniformLocation(_program, "perspective");
    }
}

void SplatShader::use(const DrawBase &draw)
{
    if (!_valid)
    {
        return;
    }

    glUseProgram(_program);
    glUniformMatrix4fv(_modelPos, 1, GL_FALSE, glm::value_ptr(draw.getModelMatrix()));
    glUniformMatrix4fv(_viewPos, 1, GL_FALSE, glm::value_ptr(_camera->getView()));
    glUniformMatrix4fv(_perspectivePos, 1, GL_FALSE, glm::value_ptr(_camera->getPerspective()));
}

bool SplatShader::valid()
{
    return _valid;
}
