#pragma once

#include "Config.hpp"
#include "ShaderBase.hpp"

class ColorfulShader : public ShaderBase
{
public:
    CLASS_PTRS(ColorfulShader)

    ColorfulShader();

    virtual bool valid() override;
};
