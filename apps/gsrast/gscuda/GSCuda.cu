#include "GSCuda.cuh"
#include <iostream>

namespace gscuda 
{
    void forward(
        std::function<char *(size_t)> geometryBuffer,
        std::function<char *(size_t)> binningBuffer, 
        std::function<char *(size_t)> imageBuffer,
        int numGaussians, int shDims, int M,
        const float *background, // CUDA vec3 
        int width, int height, 
        const float *means3D, // CUDA Gaussian positions 
        const float *shs, // CUDA Spherical harmonics 
        const float *colorsPrecomp, // Unused; precomputed colors
        const float *opacities, // CUDA per-gaussian opacity 
        const float *scales, // CUDA per-gaussian scales 
        float scaleModifier, 
        const float *rotations, // CUDA per-gaussian rotation
        const float *cov3DPrecomp, // Unused; precomputed 3D covariance matrices 
        const float *viewMatrix, // CUDA mat4 
        const float *projMatrix, // CUDA mat4: perspective * view 
        const float *camPos, // CUDA vec3 
        float tanFOVx, float tanFOVy, // for focal length calculation
        bool prefiltered, // Unused  
        float *outColor, // The outColor array 
        int *radii, // Unused 
        int *rects, // CUDA rects for fast culling 
        float *boxMin, // Unused; bounding box I think 
        float *boxMax // Unused; bounding box I think
    )
    {
        std::cout << "Would you be mad if it is unimplemented?" << std::endl;
    }
};

