#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <memory>
#include <string>
#include "WindowBase.hpp"
#include "Config.hpp"

class Window : public WindowBase
{
public:
    CLASS_PTRS(Window)

    ~Window();

    virtual bool valid() const override;
    virtual bool closed() const override;
    virtual float deltaTime() const override;
    virtual void pollEvents() override;
    virtual void swapBuffers() const override;

    virtual void clear(glm::vec4 clearColor) override;
    virtual void drawFrame() override;
    virtual void addDrawable(DrawBase::Ptr db) override;
    virtual void clearDrawables() override;

protected:
    Window(const std::string &window_name, int width, int height);

    GLFWwindow *_window;
    int _w, _h;
    bool _valid;
    float _dt;
    std::vector<DrawBase::Ptr> _drawables;
};
