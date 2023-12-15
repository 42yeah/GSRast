#include "SphereTraceWindow.hpp"
#include "Config.hpp"
#include "Window.hpp"
#include "Ellipsoid.hpp"
#include "SphereTraceShader.hpp"
#include <GLFW/glfw3.h>
#include <memory>
#include <glm/gtc/matrix_transform.hpp>


SphereTraceWindow::SphereTraceWindow() : Window("Sphere trace", DEFAULT_WINDOW_W, DEFAULT_WINDOW_H)
{
    configureFirstPersonCamera();
    _firstPersonCamera->setPosition(glm::vec3(0.0f, 0.0f, -3.0f));
    _firstPersonCamera->setSpeed(3.0f);

    _shader = std::make_shared<SphereTraceShader>(getCamera());
    _ellipsoid = std::make_shared<Ellipsoid>(_shader);
    addDrawable(_ellipsoid);

    _currentAxis = 0;
}

SphereTraceWindow::~SphereTraceWindow()
{

}

void SphereTraceWindow::pollEvents()
{
    Window::pollEvents();

    glm::vec3 ellipPos = _ellipsoid->getCenter();
    glm::vec3 ellipScale = _ellipsoid->getScale();
    if (glfwGetKey(_window, GLFW_KEY_DOWN))
    {
        ellipPos.y -= _firstPersonCamera->getSpeed() * _dt;
    }
    if (glfwGetKey(_window, GLFW_KEY_UP))
    {
        ellipPos.y += _firstPersonCamera->getSpeed() * _dt;
    }
    if (glfwGetKey(_window, GLFW_KEY_LEFT))
    {
        ellipPos.x -= _firstPersonCamera->getSpeed() * _dt;
    }
    if (glfwGetKey(_window, GLFW_KEY_RIGHT))
    {
        ellipPos.x += _firstPersonCamera->getSpeed() * _dt;
    }
    if (glfwGetKey(_window, GLFW_KEY_COMMA))
    {
        ellipPos.z -= _firstPersonCamera->getSpeed() * _dt;
    }
    if (glfwGetKey(_window, GLFW_KEY_PERIOD))
    {
        ellipPos.z += _firstPersonCamera->getSpeed() * _dt;
    }
    if (glfwGetKey(_window, GLFW_KEY_EQUAL))
    {
        ellipScale[_currentAxis] += _firstPersonCamera->getSpeed() * _dt;
    }
    if (glfwGetKey(_window, GLFW_KEY_MINUS))
    {
        ellipScale[_currentAxis] -= _firstPersonCamera->getSpeed() * _dt;
    }
    _ellipsoid->setCenter(ellipPos);
    _ellipsoid->setScale(ellipScale);
}

void SphereTraceWindow::keyCallback(int key, int scancode, int action, int mods)
{
    Window::keyCallback(key, scancode, action, mods);
    if (key == GLFW_KEY_C && action == GLFW_PRESS)
    {
        _shader->toggleCubeMode();
    }
    if (key == GLFW_KEY_0 && action == GLFW_PRESS)
    {
        _currentAxis = 0;
        std::cout << "Manipulating axis: 0" << std::endl;
    }
    if (key == GLFW_KEY_1 && action == GLFW_PRESS)
    {
        _currentAxis = 1;
        std::cout << "Manipulating axis: 1" << std::endl;
    }
    if (key == GLFW_KEY_2 && action == GLFW_PRESS)
    {
        _currentAxis = 2;
        std::cout << "Manipulating axis: 2" << std::endl;
    }
}
