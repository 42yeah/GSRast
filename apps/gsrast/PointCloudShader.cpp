#include "PointCloudShader.hpp"
#include "CameraBase.hpp"
#include "ShaderBase.hpp"
#include "WindowBase.hpp"
#include <glm/gtc/type_ptr.hpp>


PointCloudShader::PointCloudShader(CameraBase::Ptr camera) : ShaderBase("shaders/pointcloud/vertex.glsl", "shaders/pointcloud/fragment.glsl")
{
    _camera = camera;
    if (_valid)
    {
        _modelPos = glGetUniformLocation(_program, "model");
        _viewPos = glGetUniformLocation(_program, "view");
        _perspectivePos = glGetUniformLocation(_program, "perspective");
    }
}

void PointCloudShader::use(const DrawBase &draw)
{
    glUseProgram(_program);
    glUniformMatrix4fv(_modelPos, 1, GL_FALSE, glm::value_ptr(draw.getModelMatrix()));
    glUniformMatrix4fv(_viewPos, 1, GL_FALSE, glm::value_ptr(_camera->getView()));
    glUniformMatrix4fv(_perspectivePos, 1, GL_FALSE, glm::value_ptr(_camera->getPerspective()));
}

bool PointCloudShader::valid()
{
    return _valid;
}
