#pragma once 

#include <functional>
#include <glm/glm.hpp>

/**
 * gscuda is the CUDA part of the GSRast, where we try to rasterize 
 * Gaussians. It is also a drop-in replacement of CudaRasterizer. 
 * The implementation of this function will strictly follow 
 * the implementation of diff_gaussian_rasterizer. 
 */
namespace gscuda 
{
    /**
     * Drop-in replacement to rasterize points only. This is not different at all
     * than glDrawArrays(0, P, GL_POINTS). 
     */
    void forwardPoints(std::function<char *(size_t)> geometryBuffer,
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

    void preprocess(int numGaussians, int shDims, int M,
                    const glm::vec3 *means3D,
                    const glm::vec3 *scales,
                    const float scaleModifier,
                    const glm::vec4 *rotations,
                    const float *opacities,
                    const float *shs,
                    bool *clamped,
                    const float *conv3DPrecomp,
                    const float *colorsPrecomp,
                    const float *viewMatrix,
                    const float *projMatrix,
                    const glm::vec3 *camPos,
                    int width, int height,
                    float focalDist,
                    float tanFOVx, float tanFOVy,
                    int *radii,
                    glm::vec2 *means2D,
                    float *depths,
                    float *cov3Ds,
                    glm::vec3 *rgb,
                    glm::vec4 *conicOpacity,
                    const dim3 grid,
                    uint32_t *tilesTouched,
                    bool prefiltered,
                    glm::ivec2 *rects,
                    glm::vec3 boxMin,
                    glm::vec3 boxMax);

    void forward(std::function<char *(size_t)> geometryBuffer,
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
