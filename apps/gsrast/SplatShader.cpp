#include "SplatShader.hpp"
#include "ShaderBase.hpp"
#include "DrawBase.hpp"
#include "apps/gsrast/GSEllipsoids.hpp"
#include <glm/gtc/type_ptr.hpp>


SplatShader::SplatShader(CameraBase::Ptr camera) : ShaderBase("shaders/splatsRef/vertex.glsl", "shaders/splatsRef/fragment.glsl")
{
    _camera = camera;
    if (_valid)
    {
        _modelPos = glGetUniformLocation(_program, "model");
        _viewPos = glGetUniformLocation(_program, "view");
        _perspectivePos = glGetUniformLocation(_program, "perspective");
        _camPos = glGetUniformLocation(_program, "camPos");
        _frontPos = glGetUniformLocation(_program, "camFront");
    }
}

void SplatShader::use(const DrawBase &draw)
{
    if (!_valid)
    {
        return;
    }
    const glm::vec3 &camPos = _camera->getPosition();
    const glm::vec3 &camFront = _camera->getFront();

    glUseProgram(_program);
    glUniformMatrix4fv(_modelPos, 1, GL_FALSE, glm::value_ptr(draw.getModelMatrix()));
    glUniformMatrix4fv(_viewPos, 1, GL_FALSE, glm::value_ptr(_camera->getView()));
    glUniformMatrix4fv(_perspectivePos, 1, GL_FALSE, glm::value_ptr(_camera->getPerspective()));
    glUniform3f(_camPos, camPos.x, camPos.y, camPos.z);
    glUniform3f(_frontPos, camFront.x, camFront.y, camFront.z);
}

void SplatShader::use(const GSEllipsoids &ellipsoids)
{
    if (!_valid)
    {
        return;
    }

    use((const DrawBase &) ellipsoids);

}

bool SplatShader::valid()
{
    return _valid;
}
