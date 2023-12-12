#pragma once

#include <glad/glad.h>
#include <string>
#include "Config.hpp"

class WindowBase;

class ShaderBase
{
public:
    CLASS_PTRS(ShaderBase)

    ~ShaderBase();

    virtual void update(const WindowBase &window);
    virtual void use();
    virtual bool valid() = 0;

    static GLuint compile(GLuint type, const std::string &path);
    static GLuint link(GLuint vertexShader, GLuint fragmentShader);

protected:
    ShaderBase(const std::string &vertexShaderPath, const std::string &fragmentShaderPath);

    GLuint _program;
    bool _valid;
};
