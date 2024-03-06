#include "GSCuda.cuh"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <iostream>
#include <cmath>
#include <limits>
#include <vector_types.h>
#include <cooperative_groups.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cub/cub.cuh>
#include "CudaHelpers.cuh"
#include "AuxBuffer.cuh"

#define NUM_THREADS 1024
#define SCREEN_SPACE_PC_BLOCKS_W 128
#define SCREEN_SPACE_PC_BLOCKS_H 128
#define BLOCK_W 16
#define BLOCK_H 16


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
    __global__ void projectPoints(pc::GeometryState geomState,
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

    __device__ glm::mat3 quatToMat(const glm::vec4 &q)
    {
        return glm::mat3(2.0 * (q.x * q.x + q.y * q.y) - 1.0, 2.0 * (q.y * q.z + q.x * q.w), 2.0 * (q.y * q.w - q.x * q.z), // 1st column
                         2.0 * (q.y * q.z - q.x * q.w), 2.0 * (q.x * q.x + q.z * q.z) - 1.0, 2.0 * (q.z * q.w + q.x * q.y), // 2nd column
                         2.0 * (q.y * q.w + q.x * q.z), 2.0 * (q.z * q.w - q.x * q.y), 2.0 * (q.x * q.x + q.w * q.w) - 1.0); // last column
    }

    /**
     * Computes the 3D covariance matrix and store it to cov3Ds
     * cov3Ds is a massive SoA with 6 floats as its stride; it stores the upper right corner of the matrix
     */
    __device__ void computeCov3D(const glm::vec3 &scale, float scaleModifier, const glm::vec4 &rotation, float *cov3Ds)
    {
        // 1. Make scaling matrix
        glm::mat3 scalingMat = glm::mat3(1.0f);
        for (int i = 0; i < 3; i++)
        {
            scalingMat[i][i] = scaleModifier * scale[i]; // I love glm
        }

        // 2. Transfer quaternion to rotation matrix
        glm::mat3 rotMat = quatToMat(glm::normalize(rotation)); // They are normalized when they come in, so.

        // 3. According to paper, cov = R S S R
        glm::mat3 rs = rotMat * scalingMat;
        glm::mat3 sigma = rs * glm::transpose(rs);

        // 4. Sigma is symmetric, only store upper right
        // _ _ _
        // x _ _
        // x x _
        // I think there is an error here in the original SIBR code, but it doesn't matter anyway since its symmetric
        cov3Ds[0] = sigma[0][0];
        cov3Ds[1] = sigma[1][0];
        cov3Ds[2] = sigma[2][0];
        cov3Ds[3] = sigma[1][1];
        cov3Ds[4] = sigma[2][1];
        cov3Ds[5] = sigma[2][2];
    }

    __device__ glm::vec3 computeCov2D(const glm::vec3 &mean, float focalDist, float tanFOVx, float tanFOVy, const float *cov3D, const float *viewMatrix)
    {
        // Follows Eq. 29 and Eq. 31 of EWA splatting, with focal length taken into consideration.
        // 1. Transform point center to camera space (t)
        const glm::mat4 &viewMat = reinterpret_cast<const glm::mat4 &>(*viewMatrix);
        glm::vec3 t = glm::vec3(viewMat * glm::vec4(mean, 1.0f));

        float limx = 1.3f * tanFOVx;
        float limy = 1.3f * tanFOVy;
        float txtz = t.x / t.z;
        float tytz = t.y / t.z;

        t.x = glm::min(limx, glm::max(-limx, txtz)) * t.z;
        t.y = glm::min(limy, glm::max(-limy, tytz)) * t.z;

        // 2. Calculate Jacobian.
        glm::mat3 jacobian = glm::mat3(focalDist / t.z, 0.0f, (-focalDist * t.x) / (t.z * t.z),
                                       0.0f, focalDist / t.z, (-focalDist * t.y) / (t.z * t.z),
                                       0.0f, 0.0f, 0.0f);
        glm::mat3 view3x3 = glm::transpose(viewMat);
        glm::mat3 T = view3x3 * jacobian;

        // Recover the covariance matrix.
        glm::mat3 Vrk = glm::mat3(cov3D[0], cov3D[1], cov3D[2],
                                  cov3D[1], cov3D[3], cov3D[4],
                                  cov3D[2], cov3D[4], cov3D[5]);

        // J * W * Vrk * W^T * J^T
        glm::mat3 cov = glm::transpose(T) * Vrk * T;
        cov[0][0] += 0.3f;
        cov[1][1] += 0.3f;

        // ???
        return glm::vec3(cov[0][0], cov[0][1], cov[1][1]);
    }

    /**
     * getRects: straight ripoff functions from SIBR.
     * TODO: I don't really understand WTF is going on here, so I am just gonna go ahead an take it.
     */
    __forceinline__ __device__ void getRect(const glm::vec2 &p, int max_radius, glm::uvec2 &rect_min, glm::uvec2 &rect_max, dim3 grid)
    {
        rect_min = {
            (unsigned int) glm::min((int) grid.x, glm::max((int)0, (int)((p.x - max_radius) / BLOCK_W))),
            (unsigned int) glm::min((int) grid.y, glm::max((int)0, (int)((p.y - max_radius) / BLOCK_H)))
        };
        rect_max = {
            (unsigned int) glm::min((int) grid.x, glm::max((int)0, (int)((p.x + max_radius + BLOCK_W - 1) / BLOCK_W))),
            (unsigned int) glm::min((int) grid.y, glm::max((int)0, (int)((p.y + max_radius + BLOCK_H - 1) / BLOCK_H)))
        };
    }

    __forceinline__ __device__ void getRect(const glm::vec2 &p, glm::ivec2 ext_rect, glm::uvec2 &rect_min, glm::uvec2 &rect_max, dim3 grid)
    {
        rect_min = {
            (unsigned int) glm::min((int) grid.x, glm::max((int)0, (int)((p.x - ext_rect.x) / BLOCK_W))),
            (unsigned int) glm::min((int) grid.y, glm::max((int)0, (int)((p.y - ext_rect.y) / BLOCK_H)))
        };
        rect_max = {
            (unsigned int) glm::min((int) grid.x, glm::max((int)0, (int)((p.x + ext_rect.x + BLOCK_W - 1) / BLOCK_W))),
            (unsigned int) glm::min((int) grid.y, glm::max((int)0, (int)((p.y + ext_rect.y + BLOCK_H - 1) / BLOCK_H)))
        };
    }

    __global__ void preprocessCUDA(int numGaussians, int shDims, int M,
                              const glm::vec4 *means3D, // called "origPoints" in DGR
                              const glm::vec4 *scales,
                              const float scaleModifier,
                              const glm::vec4 *rotations,
                              const float *opacities,
                              const float *shs,
                              bool *clamped,
                              const float *cov3DPrecomp,
                              const float *colorsPrecomp,
                              const float *viewMatrix,
                              const float *projMatrix,
                              const glm::vec3 *camPos,
                              int width, int height,
                              float tanFOVx, float tanFOVy,
                              float focalDist,
                              int *radii,
                              glm::vec2 *means2D, // called "points_xy_image" in DGR
                              float *depths,
                              float *cov3Ds,
                              glm::vec3 *rgb,
                              glm::vec4 *conicOpacity,
                              const dim3 grid,
                              uint32_t *tilesTouched,
                              bool prefiltered,
                              glm::ivec2 *rects,
                              glm::vec3 boxMin,
                              glm::vec3 boxMax)
    {
        namespace cg = cooperative_groups;
        size_t idx = cg::this_grid().thread_rank();
        if (idx >= numGaussians)
        {
            return;
        }

        // 1. Initialize radius and touched tiles to 0.
        radii[idx] = 0;
        tilesTouched[idx] = 0;

        // 2. Cull outside gaussians.
        const glm::mat4 &projMat = reinterpret_cast<const glm::mat4 &>(*projMatrix);
        glm::vec4 pointHom = projMat * means3D[idx];
        float oneOverW = 1.0f / (0.001f + pointHom.w);
        glm::vec3 projected = oneOverW * glm::vec3(pointHom.x, pointHom.y, pointHom.z);
        if (projected.z < 0.0f || projected.z > 1.0f || projected.x < -1.3f || projected.x > 1.3f || projected.y < -1.3f || projected.y > 1.3f)
        {
            return;
        }
        // 2.1. TODO: normally there should be a bounding box check here. But we have not supplied a bounding box (well actually we had, its just meaningless)
        //            so we'll just skip here.

        // 3. If covariance matrices are precomputed, use them; otherwise we do it ourselves
        const float *cov3D = nullptr;
        if (cov3DPrecomp)
        {
            cov3D = &cov3DPrecomp[idx * 6];
        }
        else
        {
            computeCov3D(glm::vec3(scales[idx]), scaleModifier, rotations[idx], &cov3Ds[idx * 6]);
            cov3D = &cov3Ds[idx * 6];
        }

        // 4. TODO: MAGIC 1: compute screen space covariance matrix
        glm::vec3 cov = computeCov2D(means3D[idx], focalDist, tanFOVx, tanFOVy, cov3D, viewMatrix);

        // 4.1. Invert covariance (what is happening here???)
        float det = cov.x * cov.z - cov.y * cov.y;
        if (det == 0.0f)
        {
            return;
        }
        float detInv = 1.0f / det;
        glm::vec3 conic = { cov.z * detInv, -cov.y * detInv, cov.x * detInv };

        // 5. Compute extent in screen space.
        float mid = 0.5f * (cov.x + cov.z);
        float lambda1 = mid + sqrtf(glm::max(0.1f, mid * mid - det));
        float lambda2 = mid - sqrtf(glm::max(0.1f, mid * mid - det));
        float myRadius = ceil(3.0f * sqrtf(glm::max(lambda1, lambda2)));
        glm::vec2 pointImage = (glm::vec2(projected) * 0.5f + 0.5f) * glm::vec2(width, height);
        glm::uvec2 rectMin, rectMax;

        // 6. Calculate what rects are we in... maybe
        if (rects == nullptr)
        {
            getRect(pointImage, myRadius, rectMin, rectMax, grid);
        }
        else
        {
            const glm::ivec2 myRect = glm::ivec2((int) ceil(3.0f * sqrtf(cov.x)), (int) ceil(3.0f * cov.z));
            rects[idx] = myRect;
            getRect(pointImage, myRect, rectMin, rectMax, grid);
        }
        if ((rectMax.x - rectMin.x) * (rectMax.y - rectMin.y) == 0)
        {
            return;
        }

        // 7. Use colors if they are precomputed; otherwise compute SH colors
        if (!colorsPrecomp)
        {
            const glm::vec3 &result = *reinterpret_cast<const glm::vec3 *>(&shs[idx * 48]);
            rgb[idx] = 0.5f + 0.4f * result; // Just use the bare bones SH first
        }

        // 8. Set depth
        depths[idx] = projected.z;
        radii[idx] = myRadius;
        means2D[idx] = pointImage;
        // Inverted covariance and conic opacity
        conicOpacity[idx] = glm::vec4(conic.x, conic.y, conic.z, opacities[idx]);
        tilesTouched[idx] = (rectMax.x - rectMin.x) * (rectMax.y - rectMin.y);
    }

    void preprocess(int numGaussians, int shDims, int M,
                    const glm::vec4 *means3D,
                    const glm::vec4 *scales,
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
                    glm::vec3 boxMax)
    {
        preprocessCUDA<<<(numGaussians + 255) / 256, 256>>>(numGaussians, shDims, M,
                                                            means3D, scales, scaleModifier,
                                                            rotations, opacities, shs, clamped,
                                                            conv3DPrecomp, colorsPrecomp,
                                                            viewMatrix, projMatrix, camPos,
                                                            width, height, tanFOVx, tanFOVy, focalDist,
                                                            radii, means2D, depths, cov3Ds, rgb, conicOpacity,
                                                            grid, tilesTouched, prefiltered, rects,
                                                            boxMin, boxMax);
    }

    /**
     * For each Gaussian, duplicateWithKeys fills up their respective portion of
     * unsorted point keys and so on. For example, if this Gaussian takes up 4 tiles in a screen,
     * then four keys are gonna be duplicated. That's my guess.
     */
    __global__ void duplicateWithKeys(int numGaussians,
                                      const glm::vec2 *means2D,
                                      const float *depths,
                                      const uint32_t *offsets,
                                      uint64_t *gaussianKeysUnsorted,
                                      uint32_t *gaussianValuesUnsorted,
                                      int *radii,
                                      dim3 grid,
                                      glm::ivec2 *rects)
    {
        namespace cg = cooperative_groups;
        size_t idx = cg::this_grid().thread_rank();
        if (idx >= numGaussians)
        {
            return;
        }

        // We do not care for culled Gaussians
        if (radii[idx] <= 0)
        {
            return;
        }

        // Find the offset of the Gaussian. During the inclusiveSum, the offset becomes
        // 2, 3, 6, 8, ... etc, with each adding their own number of touched tiles.
        uint32_t offset = (idx == 0) ? 0 : offsets[idx - 1];
        glm::uvec2 rectMin, rectMax;
        if (rects == nullptr)
        {
            getRect(means2D[idx], radii[idx], rectMin, rectMax, grid);
        }
        else
        {
            getRect(means2D[idx], rects[idx], rectMin, rectMax, grid);
        }

        // Later during sorting:
        // Tile IDs will be in order
        // Same tile IDs will have depths properly sorted (so long as depth is positive, comparison is correct)
        for (int y = rectMin.y; y < rectMax.y; y++)
        {
            for (int x = rectMin.x; x < rectMax.x; x++)
            {
                // 0000 0000 0000 0000 TILE ID__ IS__ HERE DEPT H_IN TERP RETED _AS_ UINT 32_T ____
                uint64_t key = y * grid.x + x;
                key <<= 32;
                const uint32_t &depth = reinterpret_cast<const uint32_t &>(depths[idx]);
                key |= depth;
                gaussianKeysUnsorted[offset] = key;
                gaussianValuesUnsorted[offset] = idx;
                offset++;
            }
        }
    }

    /**
     * Find the Most Significant Bit (MSB) to
     * accelerate the CUB RadixSort. getHigherMsb is a binary search to locate the highest bit.
     */
    uint32_t getHigherMsb(uint32_t n)
    {
        int msb = sizeof(uint32_t) * 4;
        int step = msb;
        while (step > 1)
        {
            step /= 2;
            if (n >> msb)
            {
                msb += step;
            }
            else
            {
                msb -= step;
            }
        }
        if (n >> msb)
        {
            msb++;
        }
        return msb;
    }

    __global__ void identifyTileRanges(int numRendered, uint64_t *pointListKeys, glm::uvec2 *ranges)
    {
        namespace cg = cooperative_groups;
        size_t idx = cg::this_grid().thread_rank();
        if (idx >= numRendered)
        {
            return;
        }

        uint64_t key = pointListKeys[idx]; // Remember, they are sorted at this point (which means a lot of them are probably like, the same)
        uint32_t currentTile = key >> 32; // Recover the original tile
        if (idx == 0)
        {
            ranges[currentTile].x = 0;
        }
        else
        {
            // Fuck me. Don't forget to bit shift!
            uint32_t prevTile = pointListKeys[idx - 1] >> 32;
            /**
             * This only happens during the switching of tiles.
             * As in, end of the old tile, and beginning of the new.
             * Therefore, the previous tile ends at idx), and this tile starts at [idx.
             */
            if (prevTile != currentTile)
            {
                ranges[prevTile].y = idx;
                ranges[currentTile].x = idx;
            }
            if (idx == numRendered - 1)
            {
                ranges[currentTile].y = numRendered;
            }
        }
    }

    /**
     * Actually render stuffs to the screen.
     */
    __global__ void renderCUDA(const glm::uvec2 * __restrict__ ranges,
                               const uint32_t * __restrict__ pointList,
                               int width, int height,
                               const glm::vec2 * __restrict__ means2D,
                               const glm::vec3 * __restrict__ features, // "colors"
                               const glm::vec4 * __restrict__ conicOpacities,
                               float * __restrict__ finalT, // "accumAlpha"
                               uint32_t * __restrict__ nContrib,
                               const glm::vec3 * __restrict__ background,
                               float * __restrict__ outColor)
    {
        namespace cg = cooperative_groups;
        cg::thread_block block = cg::this_thread_block();

        uint32_t numHorizontalBlocks = (width + BLOCK_W - 1) / BLOCK_W;
        glm::uvec2 pixMin = glm::uvec2(block.group_index().x * BLOCK_W, block.group_index().y * BLOCK_H);
        glm::uvec2 pixMax = glm::min(pixMin + glm::uvec2(BLOCK_W, BLOCK_H), glm::uvec2(width, height));

        glm::uvec2 pix = glm::uvec2(pixMin.x + block.thread_index().x, pixMin.y + block.thread_index().y);
        uint32_t pixId = pix.y * width + pix.x;

        bool inside = pix.x < pixMax.x && pix.y < pixMax.y;
        bool done = !inside;

        const glm::uvec2 &range = ranges[block.group_index().y * numHorizontalBlocks + block.group_index().x];
        constexpr int blockSize = BLOCK_W * BLOCK_H;
        /**
         * Since each block is 16x16, but actually have like, way more Gaussians in there sometimes,
         * we will need to go for multiple rounds to process all of them.
         */
        int rounds = (range.y - range.x + blockSize - 1) / blockSize;
        int work = range.y - range.x;

        __shared__ int32_t collectedIds[blockSize];
        __shared__ glm::vec2 collectedXYs[blockSize];
        __shared__ glm::vec4 collectedConicOpacities[blockSize];
        __shared__ glm::vec3 collectedColors[blockSize];

        float accumAlpha = 1.0f;
        uint32_t contributor = 0;
        uint32_t lastContributor = 0;
        glm::vec3 color = glm::vec3(0.0f);

        /**
         * The whole block will cooperate on fetching data and putting them into shared variables (faster this way)
         * and process them, I think is what's happening.
         */
        for (int i = 0; i < rounds; i++, work -= blockSize)
        {
            /**
             * If the whole block is done, then we're out
             */
            int numDone = __syncthreads_count(done);
            if (numDone == blockSize)
            {
                break;
            }

            int progress = i * blockSize + block.thread_rank();
            if (range.x + progress < range.y)
            {
                int collId = pointList[range.x + progress];
                collectedIds[block.thread_rank()] = collId;
                collectedXYs[block.thread_rank()] = means2D[collId];
                collectedConicOpacities[block.thread_rank()] = conicOpacities[collId];
                collectedColors[block.thread_rank()] = features[collId];
            }
            /**
             * The whole block now syncs. Note, we don't really need this procedure, since we can fetch them directly from the block (I think.)
             * Is it just because it is faster this way?
             */
            block.sync();

            /**
             * We now begin processing the data the whole block just collcetively fetched.
             * Note, work at this point may be less than one full block (which means an unfull collected* array,)
             * at which point we should just stop.
             *
             * In other words, we have at most 196 Gaussians to process, in this one pixel. Let's get goin'!
             */
            for (int j = 0; !done && j < glm::min(blockSize, work); j++)
            {
                contributor++;

                const glm::vec2 &screenSpace = collectedXYs[j];
                glm::vec2 delta = screenSpace - glm::vec2(pix);
                glm::vec4 conicOpacity = collectedConicOpacities[j];
                // "The alpha decays exponentially from the Gaussian center." I've read that somewhere in the paper. May not be
                // completely the same though. Well, it was either in paper or in code.
                // TODO: I have no idea what this equation means. SIBR explanation:
                // Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
                float power = -0.5f * (conicOpacity.x * delta.x * delta.x + conicOpacity.z * delta.y * delta.y) - conicOpacity.y * delta.x * delta.y;
                if (power > 0.0f)
                {
                    continue; // ?????
                }

                // Oh wait, here it is.
                // Eq. (2) from 3D Gaussian splatting paper.
                // Obtain alpha by multiplying with Gaussian opacity
                // and its exponential falloff from mean.
                // Avoid numerical instabilities (see paper appendix).
                float alpha = glm::min(0.99f, conicOpacity.w * exp(power));
                if (alpha < 1.0f / 255.0f)
                {
                    continue;
                }
                // Test the remaining alpha and see if it makes sense to continue the blend.
                // TODO: Having questions.
                float testNewAlpha = accumAlpha * (1.0f - alpha);
                if (testNewAlpha < 0.001f)
                {
                    // No alpha left to fill; I'm out of here
                    done = true;
                    continue;
                }

                // Eq. 3 from the splatting paper.
                color += collectedColors[j] * alpha * accumAlpha;
                accumAlpha = testNewAlpha;

                lastContributor = contributor;
            }
        }

        // It is done. Write all things to output buffer
        if (inside)
        {
            finalT[pixId] = accumAlpha;
            nContrib[pixId] = lastContributor;
            outColor[pixId + width * height * 0] = color.r + accumAlpha * background->r;
            outColor[pixId + width * height * 1] = color.g + accumAlpha * background->g;
            outColor[pixId + width * height * 2] = color.b + accumAlpha * background->b;
        }
    }

    void render(const dim3 grid, const dim3 block,
                const glm::uvec2 *ranges, const uint32_t *pointList,
                int width, int height,
                const glm::vec2 *means2D,
                const glm::vec3 *colors,
                const glm::vec4 *conicOpacities,
                float *finalT, // "accumAlpha"
                uint32_t *nContrib,
                const glm::vec3 *background,
                float *outColor)
    {
        renderCUDA<<<grid, block>>>(ranges, pointList, width, height,
                                    means2D, colors, conicOpacities, finalT,
                                    nContrib, background, outColor);
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
        GeometryState geomState = GeometryState::fromChunk(geoBuffer, numGaussians);
        if (radii == nullptr)
        {
            radii = geomState.internalRadii;
        }

        dim3 tileGrid = { (unsigned int) (width + BLOCK_W - 1) / BLOCK_W, (unsigned int) (height + BLOCK_H - 1) / BLOCK_H, 1 };
        dim3 block = { BLOCK_W, BLOCK_H, 1 };

        size_t imageBufferSize = required<ImageState>(width * height);
        char *imBuffer = imageBuffer(imageBufferSize + 128);
        ImageState imState = ImageState::fromChunk(imBuffer, width * height);

        constexpr float numericMin = std::numeric_limits<float>::lowest();
        constexpr float numericMax = std::numeric_limits<float>::max();
        glm::vec3 minn = glm::vec3(numericMin, numericMin, numericMin);
        glm::vec3 maxx = glm::vec3(numericMax, numericMax, numericMax);

        // Preprocess time!
        preprocess(numGaussians, shDims, M,
                   (glm::vec4 *) means3D,
                   (glm::vec4 *) scales,
                   scaleModifier,
                   (glm::vec4 *) rotations,
                   opacities,
                   shs,
                   geomState.clamped,
                   cov3DPrecomp,
                   colorsPrecomp,
                   viewMatrix, projMatrix, (glm::vec3 *) camPos,
                   width, height,
                   focalDist,
                   tanFOVx, tanFOVy,
                   radii,
                   geomState.means2D,
                   geomState.depths,
                   geomState.cov3D,
                   geomState.rgb,
                   geomState.conicOpacity,
                   tileGrid,
                   geomState.tilesTouched,
                   prefiltered,
                   (glm::ivec2 *) rects,
                   minn, maxx);

        // Preprocessed; next up get the full list of touched tiles
        cub::DeviceScan::InclusiveSum(geomState.scanningSpace, geomState.scanSize, geomState.tilesTouched, geomState.pointOffsets, numGaussians);
        cudaMemcpy(&geomState.numRendered, &geomState.pointOffsets[numGaussians - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Pointless to go any farther
        if (geomState.numRendered == 0)
        {
            return;
        }

        // BinningState is for per-numRendered, so that is sum(sum(tiles of this gaussian touched))
        // It is (mostly) less than the total number of Gaussians due to culling
        size_t binningBufferSize = required<BinningState>(geomState.numRendered);
        char *binningBuf = binningBuffer(binningBufferSize + 128);
        BinningState binningState = BinningState::fromChunk(binningBuf, geomState.numRendered);

        // Now we need to produce the keys
        duplicateWithKeys<<<(numGaussians + 255 / 256), 256>>>(numGaussians, geomState.means2D, geomState.depths, geomState.pointOffsets,
                                                               binningState.pointListKeysUnsorted, binningState.pointListUnsorted,
                                                               radii, tileGrid, (glm::ivec2 *) rects);

        int highestBit = getHigherMsb(tileGrid.x * tileGrid.y);

        // Sort Gaussians
        cub::DeviceRadixSort::SortPairs(binningState.listSortingSpace, binningState.sortingSize,
                                        binningState.pointListKeysUnsorted, binningState.pointListKeys,
                                        binningState.pointListUnsorted, binningState.pointList,
                                        geomState.numRendered, 0, highestBit + 32);

        // Determine tile ranges
        cudaMemset(imState.ranges, 0, sizeof(glm::uvec2) * width * height);
        identifyTileRanges<<<(geomState.numRendered + 255) / 256, 256>>>(geomState.numRendered, binningState.pointListKeys, imState.ranges);

        const glm::vec3 *colorsPtr = colorsPrecomp ? (const glm::vec3 *) colorsPrecomp : geomState.rgb;
        render(tileGrid, block,
               imState.ranges, binningState.pointList,
               width, height,
               geomState.means2D, colorsPtr, geomState.conicOpacity,
               imState.accumAlpha, imState.nContrib,
               (const glm::vec3 *) background,
               outColor);
    }
};

