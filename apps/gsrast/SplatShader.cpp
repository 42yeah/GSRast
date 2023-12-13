#include "SplatShader.hpp"
#include "ShaderBase.hpp"
#include "DrawBase.hpp"
#include "apps/gsrast/GSEllipsoids.hpp"
#include <glm/gtc/type_ptr.hpp>


SplatShader::SplatShader(CameraBase::Ptr camera) : ShaderBase("shaders/splats/vertex.glsl", "shaders/splats/fragment.glsl")
{
    _camera = camera;
    if (_valid)
    {
        _modelPos = glGetUniformLocation(_program, "model");
        _viewPos = glGetUniformLocation(_program, "view");
        _perspectivePos = glGetUniformLocation(_program, "perspective");

        int positionBlockPos = glGetUniformBlockIndex(_program, "splatPosition");
        glUniformBlockBinding(_program, positionBlockPos, 0);

        int scaleBlockPos = glGetUniformBlockIndex(_program, "splatScale");
        glUniformBlockBinding(_program, scaleBlockPos, 1);
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
