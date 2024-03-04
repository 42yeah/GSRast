#include "CudaBuffer.hpp"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <memory>
#include <cuda_gl_interop.h>

InteropBuffer::InteropBuffer() : CudaBuffer<float>()
{
    _buffer = GL_NONE;
    _size = 0;
    _length = 0;
    _graphicsResource = nullptr;
    _mapped = false;
}

GLuint InteropBuffer::getBuffer() const
{
    return _buffer;
}

void InteropBuffer::allocate(int size)
{
    free();

    // Might as well use the modern version of OpenGL
    glCreateBuffers(1, &_buffer);
    glNamedBufferStorage(_buffer, size, nullptr, GL_DYNAMIC_STORAGE_BIT);
    _size = size;
    _length = (int) ceil(_size / sizeof(float));

    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&_graphicsResource, _buffer, cudaGraphicsRegisterFlagsWriteDiscard));
}

void InteropBuffer::memcpy(const float *src, int size)
{
    assert(false && "Nope. memcpy not implemented.");
}

void InteropBuffer::set(const float &src)
{
    assert(false && "Set not implemented.");
}

void InteropBuffer::resize(int size)
{
    assert(false && "Resize is not allowed since we used glNamedBufferStorage.");
}

std::unique_ptr<float[]> InteropBuffer::toHost()
{
    if (_buffer == GL_NONE)
    {
        return nullptr;
    }

    mapResources(false);
    std::unique_ptr<float[]> ret;
    ret.reset(new float[_length]);

    glGetNamedBufferSubData(_buffer, 0, _size, ret.get());
    return std::move(ret);
}

float *InteropBuffer::getPtr()
{
    if (_buffer == GL_NONE)
    {
        return nullptr;
    }
    mapResources(true);

    float *bufferCuda = nullptr;
    size_t numBytes = 0;
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void **) &bufferCuda, &numBytes, _graphicsResource));
    assert(numBytes == _size && "numBytes should definitely equal to size, but is somehow not");
    return bufferCuda;
}


void InteropBuffer::free()
{
    if (_buffer != GL_NONE)
    {
        mapResources(false);
        glDeleteBuffers(1, &_buffer);
        CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(_graphicsResource));
        _buffer = GL_NONE;
        _size = 0;
        _length = 0;
        _graphicsResource = nullptr;
        _mapped = false;
    }
}

int InteropBuffer::size() const
{
    return _size;
}

void InteropBuffer::mapResources(bool map)
{
    if (_mapped && !map)
    {
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &_graphicsResource));
        _mapped = false;
    }
    else if (!_mapped && map)
    {
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &_graphicsResource));
        _mapped = true;
    }
}
