#include "Window.hpp"
#include "DrawBase.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

bool baseInitialized = false;
int aliveWindows = 0;

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
