#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <string>
#include "WindowBase.hpp"
#include "Config.hpp"
#include "FirstPersonCamera.hpp"

class Window : public WindowBase
{
public:
    CLASS_PTRS(Window)

    virtual ~Window();

    virtual bool valid() const override;
    virtual bool closed() const override;
    virtual float deltaTime() const override;
    virtual void pollEvents() override;
    virtual void swapBuffers() const override;

    virtual void clear(glm::vec4 clearColor) override;
    virtual void drawFrame() override;
    virtual void addDrawable(DrawBase::Ptr db) override;
    virtual void clearDrawables() override;

    virtual int getWidth() const override;
    virtual int getHeight() const override;

    /**
     * Window is capable of handling a firstPersonCamera itself.
     */
    void configureFirstPersonCamera();

protected:
    Window(const std::string &window_name, int width, int height);

    static void glfwCursorPosCallback(GLFWwindow *window, double x, double y);
    static void glfwKeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);

    virtual void cursorPosCallback(double x, double y);
    virtual void keyCallback(int key, int scancode, int action, int mods);

    GLFWwindow *_window;
    int _w, _h;
    bool _valid;
    float _dt;
    double _lastInstant;
    std::vector<DrawBase::Ptr> _drawables;

    FirstPersonCamera::Ptr _firstPersonCamera;
    bool _firstPersonMode;
    struct
    {
        glm::vec2 pos;
        bool initialized;
    } _prevCursorPos;
    glm::vec2 _cursorDelta;
};
