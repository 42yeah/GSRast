#include "SamplerShader.hpp"
#include "ShaderBase.hpp"
#include "Framebuffer.hpp"
#include <glm/gtc/type_ptr.hpp>

SamplerShader::SamplerShader() : ShaderBase("shaders/sampler/vertex.glsl", "shaders/sampler/fragment.glsl")
{
    if (_valid)
    {
        _texPos = glGetUniformLocation(_program, "tex");
    }
}

void SamplerShader::use(const DrawBase &draw)
{
    if (!_valid)
    {
        return;
    }

    // TODO: this can be further simplified to a texturebase or something
    GLuint tex = dynamic_cast<const Framebuffer &>(draw).getTexture();
    glUseProgram(_program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(_texPos, 0);
}

bool SamplerShader::valid()
{
    return _valid;
}
