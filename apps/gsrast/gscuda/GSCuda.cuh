#pragma once 

#include <functional>

/**
 * gscuda is the CUDA part of the GSRast, where we try to rasterize 
 * Gaussians. It is also a drop-in replacement of CudaRasterizer. 
 * The implementation of this function will strictly follow 
 * the implementation of diff_gaussian_rasterizer. 
 */
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
        float *boxMax);

    /**
     * Drop-in replacement to rasterize points only. This is not different at all
     * than glDrawArrays(0, P, GL_POINTS). 
     */
    void forwardPoints(
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
        float *boxMax);
};
