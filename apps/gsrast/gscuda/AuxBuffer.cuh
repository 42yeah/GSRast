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
            float *outColor; // This is the temp outColor; we do buffer swaps to prevent epilepsy
            float *defaultDepth; // just a 1.0f stored in CUDA memory

            static ImageState fromChunk(char *&, int size);
        };
    }

    namespace gs
    {
        struct GeometryState
        {
            size_t scanSize;
            float *depths; // Projected depths
            char *scanningSpace;
            bool *clamped;
            int *internalRadii;
            glm::vec2 *means2D;
            float *cov3D;
            glm::vec4 *conicOpacity;
            glm::vec3 *rgb;
            size_t *pointOffsets;
            size_t *tilesTouched;

            static GeometryState fromChunk(char *&, int numGaussians);
        };

    }
}

