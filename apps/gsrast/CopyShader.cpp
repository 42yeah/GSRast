#include "CopyShader.hpp"
#include "ShaderBase.hpp"
#include "apps/gsrast/GSGaussians.hpp"


CopyShader::CopyShader() : ShaderBase("shaders/copy/vertex.glsl", "shaders/copy/fragment.glsl")
{
    if (_valid)
    {
        _widthPos = glGetUniformLocation(_program, "width");
        _heightPos = glGetUniformLocation(_program, "height");
    }
}

CopyShader::~CopyShader()
{

}

bool CopyShader::valid()
{
    return _valid;
}

void CopyShader::use(const DrawBase &draw)
{
    if (!_valid)
    {
        return;
    }

    const GSGaussians &gauss = dynamic_cast<const GSGaussians &>(draw);

    glUseProgram(_program);
    glUniform1i(_widthPos, gauss.getWidth());
    glUniform1i(_heightPos, gauss.getHeight());
}
