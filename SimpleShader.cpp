#include "SimpleShader.hpp"
#include "ShaderBase.hpp"
#include <iostream>

SimpleShader::SimpleShader() : ShaderBase(
    "shaders/simple/vertex.glsl",
    "shaders/simple/fragment.glsl")
{

}

SimpleShader::~SimpleShader()
{

}

bool SimpleShader::valid()
{
    return _valid;
}
