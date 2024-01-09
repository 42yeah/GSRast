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
     * I don't really understand WTF is going on here, so I am just gonna go ahead an take it.
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
                              const glm::vec3 *means3D, // called "origPoints" in DGR
                              const glm::vec3 *scales,
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
        glm::vec4 pointHom = projMat * glm::vec4(means3D[idx], 1.0f);
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
            computeCov3D(scales[idx], scaleModifier, rotations[idx], &cov3Ds[idx * 6]);
            cov3D = &cov3Ds[idx * 6];
        }

        // 4. MAGIC 1: compute screen space covariance matrix
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

        dim3 tileGrid = { (unsigned int) (width + BLOCK_W - 1) / BLOCK_W, (unsigned int) (height + BLOCK_H - 1) / BLOCK_H, 1 };
        dim3 block = { BLOCK_W, BLOCK_H, 1 };

        size_t imageBufferSize = required<ImageState>(width * height);
        char *imBuffer = imageBuffer(imageBufferSize);
        ImageState imState = ImageState::fromChunk(imBuffer, width * height);

        constexpr float numericMin = std::numeric_limits<float>::lowest();
        constexpr float numericMax = std::numeric_limits<float>::max();
        glm::vec3 minn = glm::vec3(numericMin, numericMin, numericMin);
        glm::vec3 maxx = glm::vec3(numericMax, numericMax, numericMax);

        // Preprocess time!
        preprocess(numGaussians, shDims, M,
                   (glm::vec3 *) means3D,
                   (glm::vec3 *) scales,
                   scaleModifier,
                   (glm::vec4 *) rotations,
                   opacities,
                   shs,
                   geoState.clamped,
                   cov3DPrecomp,
                   colorsPrecomp,
                   viewMatrix, projMatrix, (glm::vec3 *) camPos,
                   width, height,
                   focalDist,
                   tanFOVx, tanFOVy,
                   radii,
                   geoState.means2D,
                   geoState.depths,
                   geoState.cov3D,
                   geoState.rgb,
                   geoState.conicOpacity,
                   tileGrid,
                   geoState.tilesTouched,
                   prefiltered,
                   (glm::ivec2 *) rects,
                   minn, maxx);

        std::cout << "Would you be mad if it is unimplemented?" << std::endl;
    }
};

