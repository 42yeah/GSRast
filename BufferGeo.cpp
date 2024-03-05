#include "BufferGeo.hpp"
#include "DrawBase.hpp"
#include "ShaderBase.hpp"
#include <iostream>

BufferGeo::BufferGeo() : DrawBase("BufferGeo"), _shader(nullptr), _vao(GL_NONE), _vbo(GL_NONE), _numVerts(0)
{
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);
}

BufferGeo::~BufferGeo()
{
    glDeleteVertexArrays(1, &_vao);
    glDeleteBuffers(1, &_vbo);
}

void BufferGeo::configure(float *data, int numVerts, int dataSize, std::shared_ptr<ShaderBase> shader)
{
    _numVerts = numVerts;
    _shader = shader;

    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);

    glBufferData(GL_ARRAY_BUFFER, dataSize, data, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, nullptr);
}

void BufferGeo::draw()
{
    if (_shader)
    {
        _shader->use(*this);
    }
    else
    {
        std::cerr << "No shader ???" << std::endl;
    }
    glBindVertexArray(_vao);
    glDrawArrays(GL_TRIANGLES, 0, _numVerts);
}

void BufferGeo::setShader(const ShaderBase::Ptr &shader)
{
    if (shader == nullptr) { return; }
    this->_shader = shader;
}

// BufferGeo::Ptr quadPtr = nullptr;
// float quadData[] = {
//     -1.0f, -1.0f, 0.0f,
//     1.0f, -1.0f, 0.0f,
//     1.0f, 1.0f, 0.0f,
//     1.0f, 1.0f, 0.0f,
//     -1.0f, 1.0f, 0.0f,
//     -1.0f, -1.0f, 0.0f
// };
//
// BufferGeo::Ptr BufferGeo::quad(const ShaderBase::Ptr &shader)
// {
//     if (quadPtr == nullptr)
//     {
//         quadPtr = std::make_shared<BufferGeo>();
//         quadPtr->configure(quadData, 6, sizeof(quadData), nullptr);
//     }
//     quadPtr->setShader(shader);
//     return quadPtr;
// }
