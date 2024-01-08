#pragma once

#include <glm/glm.hpp>
#include <iostream>

namespace gscuda
{
    template<typename T>
    size_t required(int num)
    {
        char *fakeChunk = nullptr; // That's 0
        T::fromChunk(fakeChunk, num);
        return reinterpret_cast<size_t>(fakeChunk);
    }

    // Pointcloud rendering
    namespace pc
    {
        // GeometryState SoA is allocated per GS
        // The struct is empty because no aux buffer is required for per-Gaussian data in PC rendering
        struct GeometryState
        {
            static GeometryState fromChunk(char *&, int numGaussians);
        };

        struct ImageState
        {
            float *depth;
            float *defaultDepth; // just a 1.0f stored in CUDA memory

            static ImageState fromChunk(char *&, int size);
        };
    }
}

