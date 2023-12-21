#include "ColorfulShader.hpp"
#include "ShaderBase.hpp"


ColorfulShader::ColorfulShader() : ShaderBase("shaders/colorful/vertex.glsl", "shaders/colorful/fragment.glsl")
{

}

bool ColorfulShader::valid()
{
    return _valid;
}
