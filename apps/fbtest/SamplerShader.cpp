#include "SamplerShader.hpp"
#include "CameraBase.hpp"
#include "ShaderBase.hpp"
#include "Framebuffer.hpp"
#include <glm/gtc/type_ptr.hpp>

SamplerShader::SamplerShader(CameraBase::Ptr camera) : ShaderBase("shaders/sampler/vertex.glsl", "shaders/sampler/fragment.glsl")
{
    _camera = camera;
    if (_valid)
    {
        _texPos = glGetUniformLocation(_program, "tex");
        _viewPos = glGetUniformLocation(_program, "view");
        _perspectivePos = glGetUniformLocation(_program, "perspective");
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
    glUniformMatrix4fv(_viewPos, 1, GL_FALSE, glm::value_ptr(_camera->getView()));
    glUniformMatrix4fv(_perspectivePos, 1, GL_FALSE, glm::value_ptr(_camera->getPerspective()));
}

bool SamplerShader::valid()
{
    return _valid;
}
