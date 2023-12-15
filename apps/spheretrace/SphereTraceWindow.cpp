#include "SphereTraceWindow.hpp"
#include "Config.hpp"
#include "Window.hpp"
#include "Cube.hpp"
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
    _cube = std::make_shared<Cube>(_shader);
    addDrawable(_cube);

    _sphereCenter = glm::vec3(0.0f);
}

SphereTraceWindow::~SphereTraceWindow()
{

}

void SphereTraceWindow::pollEvents()
{
    Window::pollEvents();

    if (glfwGetKey(_window, GLFW_KEY_DOWN))
    {
        _sphereCenter.y -= _firstPersonCamera->getSpeed() * _dt;
    }
    if (glfwGetKey(_window, GLFW_KEY_UP))
    {
        _sphereCenter.y += _firstPersonCamera->getSpeed() * _dt;
    }
    if (glfwGetKey(_window, GLFW_KEY_LEFT))
    {
        _sphereCenter.x -= _firstPersonCamera->getSpeed() * _dt;
    }
    if (glfwGetKey(_window, GLFW_KEY_RIGHT))
    {
        _sphereCenter.x += _firstPersonCamera->getSpeed() * _dt;
    }
    if (glfwGetKey(_window, GLFW_KEY_COMMA))
    {
        _sphereCenter.z -= _firstPersonCamera->getSpeed() * _dt;
    }
    if (glfwGetKey(_window, GLFW_KEY_PERIOD))
    {
        _sphereCenter.z += _firstPersonCamera->getSpeed() * _dt;
    }
    _shader->setCenter(_sphereCenter);
    _cube->setModelMatrix(glm::translate(glm::mat4(1.0f), _sphereCenter));
}
