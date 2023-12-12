#include "WindowBase.hpp"
#include <GLFW/glfw3.h>

WindowBase::WindowBase() : _camera(nullptr)
{

}

CameraBase::Ptr WindowBase::getCamera() const
{
    return _camera;
}

void WindowBase::setCamera(CameraBase::Ptr camera)
{
    _camera = camera;
}
