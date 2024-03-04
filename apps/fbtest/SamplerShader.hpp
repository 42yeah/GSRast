#include "Config.hpp"
#include "ShaderBase.hpp"
#include "CameraBase.hpp"


class SamplerShader : public ShaderBase
{
public:
    CLASS_PTRS(SamplerShader)

    SamplerShader(CameraBase::Ptr camera);

    virtual void use(const DrawBase &draw) override;
    virtual bool valid() override;

protected:
    GLuint _texPos;
    GLuint _viewPos, _perspectivePos;
    CameraBase::Ptr _camera;
};
