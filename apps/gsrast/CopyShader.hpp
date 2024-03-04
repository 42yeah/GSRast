#pragma once

#include "ShaderBase.hpp"
#include "Config.hpp"


class CopyShader : public ShaderBase
{
public:
    CLASS_PTRS(CopyShader)

    CopyShader();
    ~CopyShader();

    virtual void use(const DrawBase &draw) override;

    virtual bool valid() override;

protected:
    int _widthPos, _heightPos;
};

