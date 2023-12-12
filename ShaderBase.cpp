#include "ShaderBase.hpp"
#include <iostream>
#include <fstream>
#include <sstream>


ShaderBase::~ShaderBase()
{
    if (_valid)
    {
        glDeleteProgram(_program);
    }
}

GLuint ShaderBase::compile(GLuint type, const std::string &path) {
    std::ifstream reader(path);
    if (!reader.good()) {
        std::cerr << "Bad reader to path: " << path << "?" << std::endl;
        return GL_NONE;
    }
    GLuint shader = glCreateShader(type);
    std::stringstream ss;
    ss << reader.rdbuf();
    std::string str = ss.str();
    const char *raw = str.c_str();
    glShaderSource(shader, 1, &raw, nullptr);
    glCompileShader(shader);
    GLint state = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &state);
    if (state == GL_FALSE) {
        char log[512] = {0};
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::cerr << "Failed to compile " << path << "?: " << log << std::endl;
        glDeleteShader(shader);
        return GL_NONE;
    }
    return shader;
}

GLuint ShaderBase::link(GLuint vertexShader, GLuint fragmentShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    GLint state = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &state);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    if (state == GL_FALSE) {
        char log[512] = {0};
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::cerr << "Failed to link program?: " << log << std::endl;
        glDeleteProgram(program);
        return GL_NONE;
    }
    return program;
}

void ShaderBase::use()
{
    if (_valid)
    {
        glUseProgram(_program);
    }
}

ShaderBase::ShaderBase(const std::string &vertexShaderPath, const std::string &fragmentShaderPath)
{
    GLuint vs = ShaderBase::compile(GL_VERTEX_SHADER, "shaders/simple/vertex.glsl");
    GLuint fs = ShaderBase::compile(GL_FRAGMENT_SHADER, "shaders/simple/fragment.glsl");

    if (vs == GL_NONE || fs == GL_NONE)
    {
        std::cerr << "Invalid shaders supplied?" << std::endl;
        return;
    }

    _program = ShaderBase::link(vs, fs);
    if (_program == GL_NONE)
    {
        std::cerr << "Invalid program?" << std::endl;
        return;
    }

    _valid = true;
}
