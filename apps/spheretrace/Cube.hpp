#pragma once

#include "BufferGeo.hpp"
#include "ShaderBase.hpp"
#include "Config.hpp"


class Cube : public BufferGeo
{
public:
    CLASS_PTRS(Cube)

    Cube(ShaderBase::Ptr shader);
    ~Cube();

protected:
};
