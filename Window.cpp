#include "Window.hpp"
#include "DrawBase.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <memory>
#include <map>
#include "CameraBase.hpp"
#include "FirstPersonCamera.hpp"

bool baseInitialized = false;
int aliveWindows = 0;
std::map<GLFWwindow *, Window *> reverseLookup;

Window::Window(const std::string &window_name, int width, int height) : WindowBase()
{
    if (!baseInitialized)
    {
        if (glfwInit() == GLFW_FALSE)
        {
            std::cerr << "Cannot load GLFW?" << std::endl;
            return;
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        baseInitialized = true;
    }

    _window = glfwCreateWindow(width, height, window_name.c_str(), nullptr, nullptr);
    if (!_window)
    {
        std::cerr << "No window created?" << std::endl;
        _valid = false;
        return;
    }
    aliveWindows++;
    glfwMakeContextCurrent(_window);
    if (!gladLoadGL())
    {
        std::cerr << "Cannot load GLAD?" << std::endl;
        _valid = false;
    }
    _w = width;
    _h = height;
    _valid = true;
    _lastInstant = glfwGetTime();
    _firstPersonPtr = nullptr;
    _firstPersonMode = false;
    _cursorDelta = glm::vec2(0.0f, 0.0f);
    _prevCursorPos = glm::vec2(-1.0f, -1.0f);
    reverseLookup[_window] = this;

    glEnable(GL_DEPTH_TEST);
}

Window::~Window()
{
    // delete all drawables first
    _drawables.clear();

    if (_window)
    {
        glfwDestroyWindow(_window);
        aliveWindows--;
        if (aliveWindows == 0)
        {
            glfwTerminate();
            std::cout << "No alive windows remaining. Terminating." << std::endl;
        }
    }
}

bool Window::valid() const
{
    return _valid;
}

bool Window::closed() const
{
    return valid() && glfwWindowShouldClose(_window);
}

float Window::deltaTime() const
{
    return _dt;
}

void Window::pollEvents()
{
    double thisInstant = glfwGetTime();
    _dt = (float) (thisInstant - _lastInstant);
    _lastInstant = thisInstant;

    if (_firstPersonMode)
    {
        FirstPersonCamera::Ptr fpCam = std::dynamic_pointer_cast<FirstPersonCamera>(_camera);
        if (glfwGetKey(_window, GLFW_KEY_W))
        {
            fpCam->applyMotion(fpCam->getFront() * _dt);
        }
        if (glfwGetKey(_window, GLFW_KEY_A))
        {
            fpCam->applyMotion(-fpCam->getRight() * _dt);
        }
        if (glfwGetKey(_window, GLFW_KEY_S))
        {
            fpCam->applyMotion(-fpCam->getFront() * _dt);
        }
        if (glfwGetKey(_window, GLFW_KEY_D))
        {
            fpCam->applyMotion(fpCam->getRight() * _dt);
        }
    }

    if (_camera)
    {
        _camera->update(*this);
    }

    glfwPollEvents();
}

void Window::swapBuffers() const
{
    glfwSwapBuffers(_window);
}

void Window::clear(glm::vec4 clearColor)
{
    glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Window::drawFrame()
{
    for (int i = 0; i < _drawables.size(); i++)
    {
        _drawables[i]->draw();
    }
}

void Window::addDrawable(std::shared_ptr<DrawBase> db)
{
    _drawables.push_back(db);
}

void Window::clearDrawables()
{
    _drawables.clear();
}

int Window::getWidth() const
{
    return _w;
}

int Window::getHeight() const
{
    return _h;
}

void glfwCursorPosCallback(GLFWwindow *window, double x, double y)
{
    Window *w = reverseLookup[window];
    w->cursorPosCallback(x, y);
}

void glfwKeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    Window *w = reverseLookup[window];
    w->keyCallback(key, scancode, action, mods);
}

void Window::cursorPosCallback(double x, double y)
{
    if (_prevCursorPos.x < 0.0f)
    {
        _prevCursorPos = glm::vec2(x, y);
        return;
    }
    glm::vec2 thisCursorPos = glm::vec2(x, y);
    _cursorDelta = thisCursorPos - _prevCursorPos;
    _prevCursorPos = thisCursorPos;

    if (!_camera || !_firstPersonMode)
    {
        return;
    }
    // This WILL go wrong if the camera is not FP cam.
    FirstPersonCamera::Ptr fpCam = std::dynamic_pointer_cast<FirstPersonCamera>(_camera);
    fpCam->applyDelta(_cursorDelta.x, _cursorDelta.y);
}

void Window::configureFirstPersonCamera()
{
    _firstPersonPtr = std::make_shared<FirstPersonCamera>();
    _firstPersonMode = true;
    setCamera(_firstPersonPtr);
    glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    _cursorDelta = glm::vec2(0.0f, 0.0f);
    _prevCursorPos = glm::vec2(-1.0f, -1.0f);

    glfwSetCursorPosCallback(_window, glfwCursorPosCallback);
    glfwSetKeyCallback(_window, glfwKeyCallback);
}

void Window::keyCallback(int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        _firstPersonMode = !_firstPersonMode;
        if (_firstPersonMode)
        {
            glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        else
        {
            glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
}

