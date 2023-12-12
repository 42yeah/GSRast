#pragma once

#include "ShaderBase.hpp"
#include <glad/glad.h>
#include "Config.hpp"

class SimpleShader : public ShaderBase
{
public:
    CLASS_PTRS(SimpleShader)

    SimpleShader();
    ~SimpleShader();

    virtual bool valid() override;
};
