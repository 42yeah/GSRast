#include "GSCuda.cuh"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <glm/fwd.hpp>
#include <iostream>
#include <cmath>
#include <vector_types.h>
#include <cooperative_groups.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "AuxBuffer.cuh"

#define NUM_THREADS 1024
#define SCREEN_SPACE_PC_BLOCKS_W 128
#define SCREEN_SPACE_PC_BLOCKS_H 128
#define TILE_W 16
#define TILE_H 16


namespace gscuda 
{
    template<int ch>
    __global__ void clearColor(float *outColor,
                               const float *background,
                               int width, int height)
    {
        namespace cg = cooperative_groups;
        size_t idx = cg::this_grid().thread_rank();
        if (idx >= width * height)
        {
            return;
        }
        for (int i = 0; i < ch; i++)
        {
            outColor[idx + i * width * height] = background[i];
        }
    }

    /**
     * This function projects all points to screen space (unit: pixel.)
     */
    __global__ void projectPoints(pc::GeometryState geoState,
                                  pc::ImageState imState,
                                  const float *means3D,
                                  const float *shs,
                                  int numGaussians,
                                  const float *projMatrix,
                                  int width, int height)
    {
        namespace cg = cooperative_groups;
        size_t idx = cg::this_grid().thread_rank();
        if (idx >= numGaussians)
        {
            return;
        }
        // Step 1. project the points to image plane
        glm::vec3 coord = glm::vec3(means3D[idx * 3 + 0], means3D[idx * 3 + 1], means3D[idx * 3 + 2]);
        const glm::mat4 &projMat = *reinterpret_cast<const glm::mat4 *>(projMatrix);
        glm::vec4 pointHom = projMat * glm::vec4(coord, 1.0f);
        float oneOverW = 1.0f / (pointHom.w + 0.001f);

        // Step 2. early discard points if outside spectrum
        glm::vec3 projected = glm::vec3(pointHom.x * oneOverW, pointHom.y * oneOverW, pointHom.z * oneOverW);
        if (projected.z < 0.0f || projected.z > 1.0f || projected.x < -1.0f || projected.x > 1.0f || projected.y < -1.0f || projected.y > 1.0f)
        {
            return;
        }

        // Step 3. find out corresponding pixel coordinate, and perform depth test
        glm::vec2 imagePlane = glm::vec2((0.5f + 0.5 * projected.x) * width, (0.5f + 0.5 * projected.y) * height);
        glm::ivec2 pixelCoord = glm::ivec2((glm::int32) round(imagePlane.x), (glm::int32) round(imagePlane.y));
        if (pixelCoord.x < 0 || pixelCoord.x >= width || pixelCoord.y < 0 || pixelCoord.y >= height)
        {
            return;
        }
        unsigned int depth = *reinterpret_cast<unsigned int *>(&projected.z);
        float *depthAtPixel = &imState.depth[pixelCoord.y * width + pixelCoord.x];
        unsigned int old = atomicMin(reinterpret_cast<unsigned int *>(depthAtPixel), depth);
        if (depth >= old)
        {
            return;
        }

        // Step 4. sample colors
        size_t shOffset = idx * 3 * 16; // 3 channels * 16 coefficients
        glm::vec3 color = 0.4f * glm::vec3(shs[shOffset + 0], shs[shOffset + 1], shs[shOffset + 2]) + 0.5f;

        // Step 5. at this point, only the following will happen:
        //         1. pass the atomic min: only one shall pass, since it is atomic. All others will fail
        //         2. fail the atomic min: the function is already ended
        // therefore, it's rasterizin' time.
        int colorOffset = pixelCoord.y * width + pixelCoord.x;
        imState.outColor[colorOffset + 0 * width * height] = color.r;
        imState.outColor[colorOffset + 1 * width * height] = color.g;
        imState.outColor[colorOffset + 2 * width * height] = color.b;
    }

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
                       float *boxMax)
    {
        size_t geoBufferSize = required<pc::GeometryState>(numGaussians);
        char *geoBuffer = geometryBuffer(geoBufferSize + 32);
        pc::GeometryState geometryState = pc::GeometryState::fromChunk(geoBuffer, numGaussians);
        // std::cout << "Allocating 128-bit aligned " << geoBufferSize << " bytes of memory for geometry states" << std::endl;

        size_t imageBufferSize = required<pc::ImageState>(width * height);
        char *imBuffer = imageBuffer(imageBufferSize + 32);
        pc::ImageState imState = pc::ImageState::fromChunk(imBuffer, width * height);

        // Step 1. clear the color buffer and depth buffer
        dim3 clearDim = { (unsigned int) width, (unsigned int) height, 1 };
        clearColor<3><<<clearDim, 1>>>(imState.outColor, background, width, height);

        constexpr float farthestDepth = 1.0f;
        cudaMemcpy(imState.defaultDepth, &farthestDepth, sizeof(float), cudaMemcpyHostToDevice);
        clearColor<1><<<clearDim, 1>>>(imState.depth, imState.defaultDepth, width, height);

        // std::cout << "Allocating 128-bit aligned " << imageBufferSize << " bytes of memory for image states" << std::endl;

        // Project them all into screen space (pixel coordinates).
        int numBlocks = (numGaussians + NUM_THREADS - 1) / NUM_THREADS;
        // std::cout << "Launching blocks: " << numBlocks << ", threads: " << NUM_THREADS << std::endl;
        assert(numBlocks <= 65535 && "Too many blocks (in one dimension)"); // https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)
        projectPoints<<<numBlocks, NUM_THREADS>>>(geometryState, imState, means3D, shs, numGaussians, projMatrix, width, height);
        cudaDeviceSynchronize();

        // Copy outColor from imState to the real outColor, using one copy call
        cudaMemcpy(outColor, imState.outColor, width * height * sizeof(float) * 3, cudaMemcpyDeviceToDevice);
    }

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
                 float *boxMax)
    {
        using namespace gs;
        float focalDist = height / (2.0f * tanFOVy);

        size_t geometryBufferSize = required<GeometryState>(numGaussians);
        char *geoBuffer = geometryBuffer(geometryBufferSize);
        GeometryState geoState = GeometryState::fromChunk(geoBuffer, numGaussians);
        if (radii == nullptr)
        {
            radii = geoState.internalRadii;
        }

        std::cout << "Would you be mad if it is unimplemented?" << std::endl;
    }
};

