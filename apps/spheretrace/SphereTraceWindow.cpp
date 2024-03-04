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
    glm::vec2 ellipYP = _ellipsoid->getYawPitch();
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
    if (glfwGetKey(_window, GLFW_KEY_LEFT_BRACKET) && _currentAxis < 2)
    {
        ellipYP[_currentAxis] -= _firstPersonCamera->getSpeed() * _dt;
    }
    if (glfwGetKey(_window, GLFW_KEY_RIGHT_BRACKET) && _currentAxis < 2)
    {
        ellipYP[_currentAxis] += _firstPersonCamera->getSpeed() * _dt;
    }

    _ellipsoid->setCenter(ellipPos);
    _ellipsoid->setScale(ellipScale);
    _ellipsoid->setYawPitch(ellipYP);
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
    if (glfwGetKey(_window, GLFW_KEY_P))
    {

        glm::mat4 positions = glm::mat4(1.0f, 0.0f, 0.0f, 1.0f,
                                        0.0f, 1.0f, 0.0f, 1.0f,
                                        0.0f, 0.0f, 1.0f, 1.0f,
                                        0.0f, 0.0f, 0.0f, 1.0f);
        glm::mat4 positionsInv = -positions;
        glm::mat4 projected = _camera->getPerspective() * _camera->getView() * _ellipsoid->getModelMatrix() * positions;
        glm::mat4 projInv = _camera->getPerspective() * _camera->getView() *_ellipsoid->getModelMatrix() * positionsInv;
        std::cout << projected << ", " << projInv << std::endl;
        for (int i = 0; i < 4; i++)
        {
            projected[i] = projected[i] / projected[i].w;
            projInv[i] = projInv[i] / glm::abs(projInv[i].w);
            projected[i] = 0.5f * (projected[i] - projInv[i]);
        }
        glm::vec2 u = projected[0], v = projected[1], w = projected[2];
        std::cout << u << v << w << std::endl;
        std::cout << "Their dots: " << glm::dot(glm::normalize(u), glm::normalize(v)) << ", " << glm::dot(glm::normalize(u), glm::normalize(w)) << ", " << glm::dot(glm::normalize(v), glm::normalize(w)) << std::endl;

        // Axis-aligned projection
        glm::mat4 scaleMat = glm::mat4(_ellipsoid->getScale().x, 0.0, 0.0, 0.0,
                    0.0, _ellipsoid->getScale().y, 0.0, 0.0,
                    0.0, 0.0, _ellipsoid->getScale().z, 0.0,
                    0.0, 0.0, 0.0, 1.0);
        glm::mat4 camMove = glm::mat4(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f), glm::vec4(0.0f, 1.0f, 0.0f, 0.0f), glm::vec4(0.0f, 0.0f, 1.0f, 0.0f), glm::vec4(-_camera->getPosition(), 1.0));
        glm::mat4 aaProj = _camera->getPerspective() * camMove * scaleMat * glm::mat4(1.0, 0.0, 0.0, 1.0,
                                            0.0, 1.0, 0.0, 1.0,
                                            0.0, 0.0, 1.0, 1.0,
                                            0.0, 0.0, 0.0, 1.0);
        glm::vec4 pa = aaProj[0];
        glm::vec4 pb = aaProj[1];
        glm::vec4 pc = aaProj[2];
        pa /= pa.w;
        pb /= pb.w;
        pc /= pc.w;

        glm::vec2 paa = glm::vec2(pa);
        glm::vec2 pbb = glm::vec2(pb);
        glm::vec2 pcc = glm::vec2(pc);
        std::cout << paa << pbb << pcc << std::endl;
        std::cout << glm::length(paa) << ", " << glm::length(pbb) << ", " << glm::length(pcc) << std::endl;
    }
}
