#pragma once

#include <glad/glad.h>
#include "Config.hpp"
#include <cuda_runtime.h>
#include <driver_types.h>

#define CHECK_CUDA_ERROR(A) \
A; \
cudaDeviceSynchronize(); \
if (cudaPeekAtLastError() != cudaSuccess) \
std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(cudaGetLastError()) << "?" << std::endl;

template<typename T>
class CudaBuffer
{
public:
    CLASS_PTRS(CudaBuffer)

    CudaBuffer() : _ptr(nullptr), _size(0)
    {

    }

    virtual void allocate(int size)
    {
        free();
        _size = size;
        CHECK_CUDA_ERROR(cudaMalloc(&_ptr, _size));
        CHECK_CUDA_ERROR(cudaMemset(_ptr, 0, _size));
    }

    virtual void memcpy(const T *src, int size)
    {
        if (size > _size || !_ptr)
        {
            allocate(size);
        }
        CHECK_CUDA_ERROR(cudaMemcpy(_ptr, src, size, cudaMemcpyHostToDevice));
    }

    virtual void set(const T &src)
    {
        if (_size < sizeof(T))
        {
            allocate(sizeof(T));
        }
        CHECK_CUDA_ERROR(cudaMemcpy(_ptr, &src, _size, cudaMemcpyHostToDevice));
    }

    virtual void resize(int size)
    {
        int oldSize = _size;
        int copySize = oldSize < size ? oldSize : size;
        T *newPtr = nullptr;
        CHECK_CUDA_ERROR(cudaMalloc(&newPtr, size));
        CHECK_CUDA_ERROR(cudaMemset(newPtr, 0, size));
        if (oldSize > 0)
        {
            CHECK_CUDA_ERROR(cudaMemcpy(newPtr, _ptr, copySize, cudaMemcpyDeviceToDevice));
        }
        CHECK_CUDA_ERROR(cudaFree(_ptr));
        _size = size;
        _ptr = newPtr;
    }

    virtual std::unique_ptr<T[]> toHost()
    {
        std::unique_ptr<T[]> ret = nullptr;
        if (!_ptr)
        {
            return std::move(ret);
        }
        ret.reset(new T[_size / sizeof(T)]);
        CHECK_CUDA_ERROR(cudaMemcpy(ret.get(), _ptr, _size, cudaMemcpyDeviceToHost));
        return std::move(ret);
    }

    virtual T *getPtr()
    {
        return _ptr;
    }

    virtual void free()
    {
        if (_ptr)
        {
            CHECK_CUDA_ERROR(cudaFree(_ptr));
            _ptr = nullptr;
            _size = 0;
        }
    }

    ~CudaBuffer()
    {
        free();
    }

    CudaBuffer(const CudaBuffer &another)
    {
        allocate(another._size);
        CHECK_CUDA_ERROR(cudaMemcpy(_ptr, another._ptr, _size, cudaMemcpyDeviceToDevice));
    }

    virtual int size() const
    {
        return _size;
    }

protected:
    T *_ptr;
    int _size;
};

class InteropBuffer : public CudaBuffer<float>
{
public:
    CLASS_PTRS(InteropBuffer)

    InteropBuffer();
    InteropBuffer(const InteropBuffer &) = delete;

    GLuint getBuffer() const;

    virtual void allocate(int size) override;
    virtual void memcpy(const float *src, int size) override;
    virtual void set(const float &src) override;
    virtual void resize(int size) override;
    virtual std::unique_ptr<float[]> toHost() override;
    virtual float *getPtr() override;
    virtual void free() override;
    virtual int size() const override;

protected:
    void mapResources(bool map);

    GLuint _buffer;
    int _size, _length;
    cudaGraphicsResource_t _graphicsResource;
    bool _mapped;
};
