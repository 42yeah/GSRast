#include "DebugCamera.hpp"
#include "CameraBase.hpp"
#include <glm/fwd.hpp>


DebugCamera::DebugCamera() : CameraBase()
{
    _view = glm::transpose(glm::mat4(0.87f, 0.06f, 0.47f, 0.0f,
                                     -0.04f, 0.99f, -0.05f, 0.0f,
                                     -0.47, 0.027f, 0.87f, 0.0f,
                                     0.83f, 0.42f, 4.72f, 1.0f));
    _projection = glm::transpose(glm::mat4(1.04f, 0.14f, 0.47f, 0.47f,
                                           -0.05f, 2.13f, -0.05f, -0.05f,
                                           -0.56f, 0.05f, 0.87f, 0.87f,
                                           0.98f, 0.90f, 4.7f, 4.7f));
    _eye = glm::vec3(0.83f, 0.42f, 4.72f);
}

const glm::mat4 &DebugCamera::getPerspective() const
{
    return _projection;
}

const glm::mat4 &DebugCamera::getView() const
{
    return _view;
}

const glm::vec3 &DebugCamera::getPosition() const
{
    return _eye;
}

const glm::vec3 &DebugCamera::getFront() const
{
    return _eye;
}

void DebugCamera::update(const WindowBase &window)
{
    // Does nothing
}
