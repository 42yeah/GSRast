#include "AuxBuffer.cuh"

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
        out = reinterpret_cast<T *>(offset);
        chunk = reinterpret_cast<char *>(out + size);
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

            return state;
        }
    }
}

