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
        /**
           Gaussian acceleration data structures: for now, we will get
           both ellipsoid and ellipse into our GeometryState to
           guarantee the results are indeed correct.
        */

        /**
           Mathematical ellipsoid representation
           i.e. $x^T A x + b x + c = 0$.
           This should be constructed from scaling, rotation, and
           translation. It's just the way it is.
        */
        struct MathematicalEllipsoid
        {
            glm::mat3 A;
            glm::vec3 b;
            float c;
        };

        struct MathematicalEllipse
        {
            glm::mat2 A;
            glm::vec2 b;
            float c;
            glm::vec2 eigenvalues;
            bool degenerate;
        };

        struct GeometryState
        {
            uint32_t *tilesTouched;
            size_t scanSize;
            uint32_t numRendered;
            char *scanningSpace;
            float *depths; // Projected depths
            bool *clamped;
            int *internalRadii;
            glm::vec2 *means2D;
            float *cov3D;
            glm::vec4 *conicOpacity;
            glm::vec3 *rgb;
            uint32_t *pointOffsets;

            // DS required for ellipsoid projection approximation.
            MathematicalEllipsoid *ellipsoids;
            MathematicalEllipse *ellipses;

            static GeometryState fromChunk(char *&, int numGaussians);
        };

        struct ImageState
        {
            glm::uvec2 *ranges;
            uint32_t *nContrib;
            float *accumAlpha;

            static ImageState fromChunk(char *&, int size);
        };

        struct BinningState
        {
            uint64_t *pointListKeysUnsorted;
            uint64_t *pointListKeys;
            uint32_t *pointListUnsorted;
            uint32_t *pointList;

            size_t sortingSize;
            char *listSortingSpace;

            static BinningState fromChunk(char *&, int size);
        };

    }
}

