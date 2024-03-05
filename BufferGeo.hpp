#pragma once

#include "DrawBase.hpp"
#include "ShaderBase.hpp"
#include <glad/glad.h>
#include "Config.hpp"

class BufferGeo : public DrawBase
{
public:
    CLASS_PTRS(BufferGeo)

    BufferGeo();
    virtual ~BufferGeo();

    virtual void configure(float *data, int numVerts, int dataSize, ShaderBase::Ptr shader);

    virtual void draw() override;
    virtual void setShader(const ShaderBase::Ptr &shader);

protected:
    ShaderBase::Ptr _shader;
    GLuint _vao, _vbo;
    int _numVerts;
};
