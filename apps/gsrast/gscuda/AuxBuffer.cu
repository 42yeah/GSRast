#include "AuxBuffer.cuh"
#include <cub/device/device_scan.cuh>


namespace gscuda
{
    /**
     * We need to align because CUDA **WILL** complain 
     * if the memory is not properly aligned to - I think at least 16 bytes?
     * chunk is both the input and output variable.
     * out is the output pointer obtained from said chunk.
     */
    template<typename T>
    void obtain(char *&chunk, T *&out, size_t size, size_t align)
    {
        assert(align >= 16 && "CUDA requires an alignment of at least 16 bytes");
        size_t offset = reinterpret_cast<size_t>(chunk);
        size_t aligned = align * ((offset + align - 1) / align);
        out = reinterpret_cast<T *>(aligned);
        chunk = reinterpret_cast<char *>(aligned + size);
    }

    namespace pc
    {
        GeometryState GeometryState::fromChunk(char *&chunk, int numGaussians)
        {
            GeometryState state;
            return state;
        }

        ImageState ImageState::fromChunk(char *&chunk, int size)
        {
            ImageState state;
            obtain(chunk, state.depth, sizeof(float) * size, 128);
            obtain(chunk, state.outColor, sizeof(float) * size * 3, 128);
            obtain(chunk, state.defaultDepth, sizeof(float), 128);

            return state;
        }
    }

    namespace gs
    {
        GeometryState GeometryState::fromChunk(char *&chunk, int numGaussians)
        {
            GeometryState state;
            obtain(chunk, state.depths, sizeof(float) * numGaussians, 128);
            obtain(chunk, state.clamped, sizeof(bool) * numGaussians * 3, 128);
            obtain(chunk, state.internalRadii, sizeof(float) * numGaussians, 128);
            obtain(chunk, state.means2D, sizeof(glm::vec2) * numGaussians, 128);
            obtain(chunk, state.cov3D, 6 * sizeof(float) * numGaussians, 128); // Upper-right corner of the matrix (because it's symmetric)
            obtain(chunk, state.conicOpacity, sizeof(glm::vec4) * numGaussians, 128);
            obtain(chunk, state.rgb, sizeof(glm::vec3) * numGaussians, 128);
            obtain(chunk, state.tilesTouched, sizeof(size_t) * numGaussians, 128);
            cub::DeviceScan::InclusiveSum(nullptr, state.scanSize, state.tilesTouched, state.tilesTouched, numGaussians);
            obtain(chunk, state.scanningSpace, state.scanSize, 128);
            obtain(chunk, state.pointOffsets, sizeof(size_t) * numGaussians, 128);

            return state;
        }
    }
}

