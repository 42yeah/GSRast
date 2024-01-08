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
        std::cout << "To allocate: " << size << " (" << (void *) size << ")" << std::endl;
        size_t offset = reinterpret_cast<size_t>(chunk);
        size_t aligned = align * ((offset + align - 1) / align);
        out = reinterpret_cast<T *>(offset);
        chunk = reinterpret_cast<char *>(out + size);

        std::cout << "Offset: " << (void *) (offset) << ", aligned: " << (void *) (aligned) << ", out: " << (void *) (out) << ", chunk: " << (void *) (chunk) << std::endl;

    }

 //    template <typename T>
	// static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	// {
	// 	std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
	// 	ptr = reinterpret_cast<T*>(offset);
	// 	chunk = reinterpret_cast<char*>(ptr + count);
	// }
    
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
            obtain(chunk, state.defaultDepth, sizeof(float), 128);

            return state;
        }
    }
}

