#include "Config.hpp"
#include "ShaderBase.hpp"


class SamplerShader : public ShaderBase
{
public:
    CLASS_PTRS(SamplerShader)

    SamplerShader();

    virtual void use(const DrawBase &draw) override;
    virtual bool valid() override;

protected:
    GLuint _texPos;
};
