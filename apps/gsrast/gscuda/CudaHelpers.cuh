#pragma once

#include <memory>
#include <cuda_runtime.h>

namespace gscuda
{
    /**
     * Peek at a CUDA buffer. Don't repeatedly call this function because it **will** be slow.
     */
    template<typename T>
    std::unique_ptr<T[]> peek(const void *buf, int size)
    {
        std::unique_ptr<T[]> ptr(new T[(size + sizeof(T) - 1) / sizeof(T)]);
        cudaMemcpy(ptr.get(), buf, size, cudaMemcpyDeviceToHost);
        return ptr;
    }
}

