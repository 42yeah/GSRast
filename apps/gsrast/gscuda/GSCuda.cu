#include "GSCuda.cuh"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <glm/common.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <cmath>
#include <glm/matrix.hpp>
#include <limits>
#include <vector_types.h>
#include <cooperative_groups.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cub/cub.cuh>
#include "AuxBuffer.cuh"
#include <limits>


#define NUM_THREADS 1024
#define SCREEN_SPACE_PC_BLOCKS_W 128
#define SCREEN_SPACE_PC_BLOCKS_H 128
#define BLOCK_W 16
#define BLOCK_H 16
#define ADAPTIVE_FUNC_SIZE 5
#define IMPACT_ALPHA 0.2f

#define TEST_TILE(xx, yy) (block.group_index().x == xx && block.group_index().y == yy \
                         && block.thread_rank() == 0)


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

    __host__ __device__ glm::mat3 quatToMat(const glm::vec4 &q)
    {
        return glm::mat3(2.0 * (q.x * q.x + q.y * q.y) - 1.0, 2.0 * (q.y * q.z + q.x * q.w), 2.0 * (q.y * q.w - q.x * q.z), // 1st column
                         2.0 * (q.y * q.z - q.x * q.w), 2.0 * (q.x * q.x + q.z * q.z) - 1.0, 2.0 * (q.z * q.w + q.x * q.y), // 2nd column
                         2.0 * (q.y * q.w + q.x * q.z), 2.0 * (q.z * q.w - q.x * q.y), 2.0 * (q.x * q.x + q.w * q.w) - 1.0); // last column
    }

    void quatToMatHost(float *mat, const float *q)
    {
        glm::mat3 out = quatToMat(reinterpret_cast<const glm::vec4 &>(*q));
        memcpy(mat, &out, sizeof(glm::mat3));
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

    __device__ void blitRect(float *compositeBuffer, int width, int height,
                             int x, int y, int w, int h,
                             glm::vec3 color)
    {
        w = glm::min(w, width);
        h = glm::min(h, height);
        const size_t stride = width * height;
        for (int yy = 0; yy < h; yy++)
        {
            for (int xx = 0; xx < w; xx++)
            {
                if (x + xx < 0 || x + xx >= width || y + yy < 0 || y + yy >= height)
                {
                    continue;
                }
                int idx = (y + yy) * width + (x + xx);
                compositeBuffer[idx + stride * 0] = color.r;
                compositeBuffer[idx + stride * 1] = color.g;
                compositeBuffer[idx + stride * 2] = color.b;
            }
        }
    }


    /**
       Debug kernel to draw a simple rectangle.
       *Extremely* low efficiency; remember to remove this upon
       successful debug.
    */
    __global__ void blitRectKernel(float *compositeBuffer, int width,
                                   int height, int x, int y, int w,
                                   int h, glm::vec3 color)
    {
        blitRect(compositeBuffer, width, height, x, y, w, h, color);
    }

    void blitRectHost(float *compositeBuffer, int width, int height,
                      int x, int y, int w, int h, glm::vec3 color)
    {
        blitRectKernel<<<1, 1>>>(compositeBuffer, width,
                                 height, x, y, w, h, color);
    }
    

    __host__ __device__ void ellipsoidFromGaussian(gs::MathematicalEllipsoid &ellip,
                                                   const glm::mat3 &rot,
                                                   const glm::vec4 &scl,
                                                   const glm::vec4 &center)
    {
        // Some minor memory speed optimizations
        memset(&ellip.A, 0, sizeof(glm::mat3));
        glm::vec3 ratioX = 0.5f * rot[0] / scl.x;
        glm::vec3 ratioY = 0.5f * rot[1] / scl.y;
        glm::vec3 ratioZ = 0.5f * rot[2] / scl.z;
        ellip.A += glm::outerProduct(ratioX, ratioX);
        ellip.A += glm::outerProduct(ratioY, ratioY);
        ellip.A += glm::outerProduct(ratioZ, ratioZ);
        glm::vec3 prod = ellip.A * glm::vec3(center);
        ellip.b = -2.0f * prod;
        ellip.c = glm::dot(glm::vec3(center), prod) - 1.0f;

        // Normalize the lot - we will need an efficient way for the
        // normalization. Normalization is required because numerical
        // instabilities will fuck our ellipsoid three times over (and
        // therefore, the projected ellipse.) One way, obviously, is
        // to use double precision; however, are we really stepping
        // that low? So without further ado, three methods to prevent
        // instabilities:
        // 1. Finding the minimum value and normalizing it to 1. This
        // prevents the minimum value from degenerating.
        float minVal = std::numeric_limits<float>::max();
        float maxVal = std::numeric_limits<float>::lowest();
        float avgVal = 0.0f;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                float fabsA = fabs(ellip.A[i][j]);
                minVal = minVal > fabsA ? fabsA : minVal;
                maxVal = maxVal < fabsA ? fabsA : maxVal;
                avgVal += fabsA;
            }
            float fabsB = fabs(ellip.b[i]);
            avgVal += fabsB;
            minVal = minVal > fabsB ? fabsB : minVal;
            maxVal = maxVal < fabsB ? fabsB : maxVal;
        }
        float fabsC = fabs(ellip.c);
        avgVal += fabsC;
        minVal = minVal > fabsC ? fabsC : minVal;
        maxVal = maxVal < fabsC ? fabsC : maxVal;
        avgVal = sqrtf(avgVal / 13.0f);

        float scale = -1.0f / ellip.c;
        ellip.A *= scale; // make the ellipses slightly bigger
        ellip.b *= scale;
        ellip.c *= scale;
    }


    void ellipsoidFromGaussianHost(gs::MathematicalEllipsoid *ellip,
                                   const float *rot, // mat3,
                                   const float *scl, // vec4,
                                   const float *center) // vec4
    {
        ellipsoidFromGaussian(*ellip,
                              reinterpret_cast<const glm::mat3 &>(*rot),
                              reinterpret_cast<const glm::vec4 &>(*scl),
                              reinterpret_cast<const glm::vec4 &>(*center));
    }

    /**
       This function projects a 3D ellipsoid into an ellipse, given
       the input transformed ellipsoid. The code is mostly inspired
       from the GeometricTools:PerspectiveProjectionEllipsoid.pdf
    */
    __host__ __device__ void projectEllipsoid(gs::MathematicalEllipse &ellipse,
                                              const gs::MathematicalEllipsoid &ellipsoid,
                                              const glm::vec3 &camPos,
                                              const glm::mat3 &planeAxes,
                                              const float projectedDistance)
    {
        glm::vec3 AE = ellipsoid.A * camPos;
        float qFormEAE = glm::dot(camPos, AE);
        float dotBE = glm::dot(ellipsoid.b, camPos);
        float quadE = 4.0f * (qFormEAE + dotBE + ellipsoid.c);
        glm::vec3 Bp2AE = ellipsoid.b + 2.0f * AE;
        glm::mat3 M = glm::outerProduct(Bp2AE, Bp2AE) - quadE * ellipsoid.A;

        // Compute projected coeffs.
        glm::vec3 Mu = M * planeAxes[1];
        glm::vec3 Mv = M * planeAxes[2];
        glm::vec3 Mn = M * planeAxes[0];

        float twoN = 2.0f * projectedDistance;

        ellipse.A[0][0] = glm::dot(planeAxes[1], Mu);
        ellipse.A[0][1] = glm::dot(planeAxes[1], Mv);
        ellipse.A[1][0] = ellipse.A[0][1];
        ellipse.A[1][1] = glm::dot(planeAxes[2], Mv);
        ellipse.b[0] = twoN * glm::dot(planeAxes[1], Mn);
        ellipse.b[1] = twoN * glm::dot(planeAxes[2], Mn);
        ellipse.c = projectedDistance * projectedDistance * glm::dot(planeAxes[0], Mn);

        // Solve for eigenvectors & eigenvalues.
        const float aPlusd = ellipse.A[0][0] + ellipse.A[1][1];
        const float delta = (aPlusd * aPlusd) - 4.0f * (ellipse.A[0][0] * ellipse.A[1][1] -
                                                         ellipse.A[0][1] * ellipse.A[1][0]);
        if (delta < 0.0f)
        {
            // the ellipse is invalid.
            ellipse.degenerate = true;
            return;
        }

        float lambda1 = 0.5f * ((aPlusd) + sqrtf(delta));
        float lambda2 = 0.5f * ((aPlusd) - sqrtf(delta));
        if (lambda1 == 0.0f ||
            lambda2 == 0.0f)
        {
            // Oops, time to bail
            ellipse.degenerate = true;
            return;
        }
        ellipse.degenerate = false;
        ellipse.eigenvalues = glm::vec2(lambda1, lambda2);

        // printf("proj dist: %f, Mn: %f %f %f, planeAxes[0]: %f %f %f, \
        //        planeAxes[1]: %f %f %f, planeAxes[2]: %f %f %f\n",
        //        projectedDistance, Mn.x, Mn.y, Mn.z,
        //        planeAxes[0].x, planeAxes[0].y, planeAxes[0].z,
        //        planeAxes[1].x, planeAxes[1].y, planeAxes[1].z,
        //        planeAxes[2].x, planeAxes[2].y, planeAxes[2].z);
    }

    void projectEllipsoidHost(gs::MathematicalEllipse *ellipse,
                              const gs::MathematicalEllipsoid *ellipsoid,
                              const float *camPos, // vec3
                              const float *planeAxes, // mat3
                              const float projectedDistance)
    {
        projectEllipsoid(*ellipse, *ellipsoid,
                         reinterpret_cast<const glm::vec4 &>(*camPos),
                         reinterpret_cast<const glm::mat3 &>(*planeAxes),
                         projectedDistance);
    }

    /**
     * getRects: straight ripoff functions from SIBR.
     * TODO: I don't really understand WTF is going on here, so I am just gonna go ahead an take it.
     */
    __device__ void getRect(const glm::vec2 &p, int max_radius, glm::uvec2 &rect_min, glm::uvec2 &rect_max, dim3 grid)
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

    __device__ void getRect(const glm::vec2 &p, glm::ivec2 ext_rect, glm::uvec2 &rect_min, glm::uvec2 &rect_max, dim3 grid)
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

    /**
       Use this function to obtain the extent of an ellipse. Then, the
       extent can be fed directly into getRect() to obtain the
       bounding rect.
    */
    __device__ void getEllipseExtent(const glm::vec4 &center,
                                     const glm::vec4 &scale,
                                     const glm::mat3 &rotate,
                                     const glm::mat4 &projMat,
                                     int width, int height,
                                     glm::uvec2 &extent)
    {
        glm::mat4 units(scale.x, 0.0f, 0.0f, 1.0f,
                        0.0f, scale.y, 0.0f, 1.0f,
                        0.0f, 0.0f, scale.z, 1.0f,
                        0.0f, 0.0f, 0.0f, 1.0f);
        glm::mat4 goodRot(rotate);
        goodRot[3][3] = 1.0f;
        units = goodRot * units;
        for (int i = 0; i < 4; i++)
        {
            units[i] += center;
            units[i].w = 1.0f;
        }
        units = projMat * units;

        glm::vec2 semiAxes(0.0f);
        for (int i = 0; i < 4; i++)
        {
            float oneOverW = 1.0f / (0.001f + units[i].w);
            units[i] *= oneOverW;
        }
        for (int i = 0; i < 3; i++)
        {
            glm::vec2 semi = units[i] - units[3];
            float absSemiX = fabs(semi.x);
            float absSemiY = fabs(semi.y);
            semiAxes.x = glm::max(semiAxes.x, absSemiX);
            semiAxes.y = glm::max(semiAxes.y, absSemiY);
        }
        semiAxes = 2.0f * semiAxes;
        extent = { semiAxes.x * width, semiAxes.y * height };
    }

    __global__ void preprocessCUDA(int numGaussians, int shDims, int M,
                                   const glm::vec4 *means3D, // called "origPoints" in DGR
                                   const glm::vec4 *scales, // "DGR" refers to diff-gaussian-rasterization
                                   const float scaleModifier,
                                   const glm::vec4 *rotations,
                                   const float *opacities,
                                   const float *shs,
                                   bool *clamped,
                                   const float *cov3DPrecomp,
                                   const float *colorsPrecomp,
                                   const float *viewMatrix,
                                   const float *perspectiveMatrix,
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
                                   gs::MathematicalEllipsoid *ellipsoids,
                                   gs::MathematicalEllipse *ellipses,
                                   const dim3 grid,
                                   uint32_t *tilesTouched,
                                   bool prefiltered,
                                   glm::ivec2 *rects,
                                   glm::vec3 boxMin,
                                   glm::vec3 boxMax,
                                   float *compositeBuffer,
                                   ForwardParams params)
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
        depths[idx] = projected.z;

        // 3. If covariance matrices are precomputed, use them; otherwise we do it ourselves
        if (!params.ellipseApprox)
        {
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

                if (idx == params.selected)
                {
                    glm::uvec2 rectSpan(rectMax.x - rectMin.x + 1,
                                        rectMax.y - rectMin.y + 1);
                    blitRect(compositeBuffer,
                             width, height,
                             rectMin.x * BLOCK_W,
                             rectMin.y * BLOCK_H,
                             rectSpan.x * BLOCK_W,
                             rectSpan.y * BLOCK_H,
                             glm::vec3(1.0f, 1.0f, 0.0f));
                    blitRect(compositeBuffer,
                             width, height,
                             pointImage.x - myRect.x,
                             pointImage.y - myRect.y,
                             myRect.x * 2, myRect.y * 2, glm::vec3(1.0f, 0.0f, 0.0f));
                }
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
            radii[idx] = myRadius;
            means2D[idx] = pointImage;
            // Inverted covariance and conic opacity
            conicOpacity[idx] = glm::vec4(conic.x, conic.y, conic.z, opacities[idx]);
            tilesTouched[idx] = (rectMax.x - rectMin.x) * (rectMax.y - rectMin.y);
        }
        else
        {
            glm::mat3 normAxes = glm::mat3(viewMatrix[2], viewMatrix[6], viewMatrix[10],
                                           viewMatrix[0], viewMatrix[4], viewMatrix[8],
                                           viewMatrix[1], viewMatrix[5], viewMatrix[9]);
            // Please, god, have mercy...
            // printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n",
            //     viewMatrix[0], viewMatrix[1], viewMatrix[2], viewMatrix[3],
            //     viewMatrix[4], viewMatrix[5], viewMatrix[6], viewMatrix[7],
            //     viewMatrix[8], viewMatrix[9], viewMatrix[10], viewMatrix[11],
            //     viewMatrix[12], viewMatrix[13], viewMatrix[14], viewMatrix[15]);

            // 2. define image plane constant
            glm::vec3 planeCenter = *camPos + params.ellipseApproxFocalDist * normAxes[0];
            const float planeConstant = glm::dot(planeCenter, normAxes[0]);
            const float projectedDistance = planeConstant -
                glm::dot(*camPos, normAxes[0]);

            // 3. turn the ellipsoid into coefficients.
            gs::MathematicalEllipsoid ellipsoid;
            glm::mat3 rotMat = quatToMat(rotations[idx]);
            ellipsoidFromGaussian(ellipsoid, rotMat,
                                  scales[idx],
                                  means3D[idx]);

            gs::MathematicalEllipse ellipse;
            projectEllipsoid(ellipse, ellipsoid, *camPos,
                             normAxes, projectedDistance);
            if (ellipse.degenerate)
            {
                return;
            }

            // We also need to set rects. However, rects are
            // relatively easy to configure in our case; since all we
            // need to do is get to know the two axes of the
            // ellipse. This still remains to be seen though.
            ellipsoids[idx] = ellipsoid;
            ellipses[idx] = ellipse;

            // Obtain semi-axes for coarse ellipse rect
            glm::vec2 pointImage = (glm::vec2(projected) * 0.5f + 0.5f) * glm::vec2(width, height);

            glm::uvec2 myRect, rectMin, rectMax;
            getEllipseExtent(means3D[idx], scales[idx], rotMat, projMat,
                             width, height, myRect);
            getRect(pointImage, myRect, rectMin, rectMax, grid);
            size_t numTiles = (rectMax.x - rectMin.x) * (rectMax.y - rectMin.y);
            if (numTiles == 0)
            {
                return;
            }

            if (!colorsPrecomp)
            {
                const glm::vec3 &result = *reinterpret_cast<const glm::vec3 *>(&shs[idx * 48]);
                rgb[idx] = 0.5f + 0.4f * result; // Just use the bare bones SH first
            }

            rects[idx] = myRect;
            tilesTouched[idx] = numTiles;
            radii[idx] = 1;

            // Since conic & opacity is useless, might as well use
            // them to transfer some debug data.
            if (idx == params.selected)
            {
                glm::uvec2 rectSpan(rectMax.x - rectMin.x + 1,
                                    rectMax.y - rectMin.y + 1);
                blitRect(compositeBuffer,
                         width, height,
                         rectMin.x * BLOCK_W,
                         rectMin.y * BLOCK_H,
                         rectSpan.x * BLOCK_W,
                         rectSpan.y * BLOCK_H,
                         glm::vec3(1.0f, 1.0f, 0.0f));
                blitRect(compositeBuffer,
                         width, height,
                         pointImage.x - myRect.x,
                         pointImage.y - myRect.y,
                         myRect.x * 2, myRect.y * 2, glm::vec3(1.0f, 0.0f, 0.0f));

                conicOpacity[idx].x = myRect.x;
                conicOpacity[idx].y = myRect.y;
            }
            conicOpacity[idx].a = opacities[idx];
            means2D[idx] = pointImage;
        }
    }

    /**
       Orthonormalize given a set of axes.
       this is used in complement of the `completeAxes` below.
    */
    __host__ __device__ void orthonormalize(glm::vec3 *axes)
    {
        for (int i = 1; i < 3; i++)
        {
            for (int j = 0; j < i; j++)
            {
                float dot = glm::dot(axes[i], axes[j]);
                axes[i] -= axes[j] * dot;
            }
            axes[i] = glm::normalize(axes[i]);
        }
    }


    __host__ __device__ void completeAxes(float *axes)
    {
        glm::vec3 *axesVec3 = reinterpret_cast<glm::vec3 *>(axes);
        const glm::vec3 &i = axesVec3[0];
        if (fabs(i.x) > fabs(i.y))
        {
            axesVec3[1] = glm::vec3(-i.z, 0.0f, i.x);
        }
        else
        {
            axesVec3[1] = glm::vec3(0.0, i.z, -i.y);
        }
        axesVec3[2] = glm::cross(i, axesVec3[1]);
        orthonormalize(axesVec3);
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
                    const float *perspectiveMatrix,
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
                    gs::MathematicalEllipsoid *ellipsoids,
                    gs::MathematicalEllipse *ellipses,
                    const dim3 grid,
                    uint32_t *tilesTouched,
                    bool prefiltered,
                    glm::ivec2 *rects,
                    glm::vec3 boxMin,
                    glm::vec3 boxMax,
                    float *compositeBuffer,
                    ForwardParams params)
    {
        preprocessCUDA<<<(numGaussians + 255) / 256, 256>>>(numGaussians, shDims, M,
                                                            means3D, scales, scaleModifier,
                                                            rotations, opacities, shs, clamped,
                                                            conv3DPrecomp, colorsPrecomp,
                                                            viewMatrix, perspectiveMatrix, projMatrix,
                                                            camPos, width, height,
                                                            tanFOVx, tanFOVy, focalDist,
                                                            radii, means2D, depths, cov3Ds, rgb, conicOpacity,
                                                            ellipsoids, ellipses, grid,
                                                            tilesTouched, prefiltered, rects,
                                                            boxMin, boxMax, compositeBuffer, params);
    }

    /**
     * For each Gaussian, duplicateWithKeys fills up their respective portion of
     * unsorted point keys and so on. For example, if this Gaussian takes up 4 tiles in a screen,
     * then four keys are gonna be duplicated. That's my guesss.
     */
    __global__ void duplicateWithKeys(int numGaussians,
                                      const glm::vec2 *means2D,
                                      const float *depths,
                                      const uint32_t *offsets,
                                      uint64_t *gaussianKeysUnsorted,
                                      uint32_t *gaussianValuesUnsorted,
                                      int *radii,
                                      dim3 grid,
                                      glm::ivec2 *rects,
                                      ForwardParams params)
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

                // Only count depth if `adaptiveOIT` is not set within
                // flags (we will construct the function manually
                // during rendering for each block)
                if (!params.adaptiveOIT)
                {
                    key <<= 32;
                    const uint32_t &depth = reinterpret_cast<const uint32_t &>(depths[idx]);
                    key |= depth;
                }

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

    __global__ void identifyTileRanges(int numRendered, uint64_t *pointListKeys,
                                       glm::uvec2 *ranges, ForwardParams params)
    {
        namespace cg = cooperative_groups;
        size_t idx = cg::this_grid().thread_rank();
        if (idx >= numRendered)
        {
            return;
        }

        uint64_t key = pointListKeys[idx]; // Remember, they are
                                           // sorted at this point
                                           // (which means a lot of
                                           // them are probably like,
                                           // the same)
        size_t shift = params.adaptiveOIT ? 0 : 32;
        uint32_t currentTile = key >> shift; // Recover the original tile
        if (idx == 0)
        {
            ranges[currentTile].x = 0;
        }
        else
        {
            // Fuck me. Don't forget to bit shift!
            uint32_t prevTile = pointListKeys[idx - 1] >> shift;
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

    __global__ void compositeCUDA(int width, int height, float *bottom, float *top)
    {
        namespace cg = cooperative_groups;
        int idx = cg::this_grid().thread_rank();

        // Trivial composition
        const size_t stride = width * height;
        bottom[idx + stride * 0] = glm::min(bottom[idx + stride * 0] + top[idx + stride * 0], 1.0f);
        bottom[idx + stride * 1] = glm::min(bottom[idx + stride * 1] + top[idx + stride * 1], 1.0f);
        bottom[idx + stride * 2] = glm::min(bottom[idx + stride * 2] + top[idx + stride * 2], 1.0f);
    }

    void composite(int width, int height, float *bottom, float *top)
    {
        dim3 blocks = { (uint32_t) width, (uint32_t) height, 1 };
        compositeCUDA<<<blocks, 1>>>(width, height, bottom, top);
    }

    inline __device__ float fastcos(float x)
    {
        constexpr float tp = 1.0f /(2.0f * 3.14f);
        x *= tp;
        x -= 0.25f + std::floor(x + 0.25f);
        x *= 16.0f * (std::abs(x) - 0.5f);
        return x;
    }

    __device__ float approxNorm2(float x)
    {
        if (x >= 2.93f)
        {
            return 0.0f;
        }
        // float sqrted = sqrtf(x);
        float sqrted = (1.0f + x) / 2.0f;
        float a = (2.0f + fastcos(sqrted)) * 0.33f;
        return a * a * a;
    }

    /**
       Initialize the adaptive function.
    */
    __host__ __device__ void initAdaptiveF(
        MiniNode *nodes, size_t numNodes, int &head, int &tail)
    {
        for (int i = 0; i < numNodes; i++)
        {
            nodes[i].prev = i - 1;
            nodes[i].next = i + 1 >= numNodes ? -1 : i + 1;
            nodes[i].depth = FLT_MAX;
            nodes[i].id = -1;
        }
        head = 0;
        tail = numNodes - 1;
    }


    /**
       Insert one member into the adaptive visibility function.
       The visibility function will keep its order, so relax.
    */
    __host__ __device__ void insertAdaptiveF(
        MiniNode *nodes, size_t numNodes,
        float depth, uint32_t id, float alpha, const glm::vec3 &color,
        int &head, int &tail)
    {
        if (alpha < IMPACT_ALPHA)
        {
            // We only keep record of the "heavy hitters".
            // TODO: aliasing problems will arise from this. Maybe we
            // only discard them when the linked list is utterly empty.
            return;
        }

        int nodeIdx = head;
        while (nodeIdx != -1 && nodes[nodeIdx].depth < depth)
        {
            nodeIdx = nodes[nodeIdx].next;
        }
        if (nodeIdx == -1)
        {
            if (alpha - nodes[tail].alpha > IMPACT_ALPHA)
            {
                nodes[tail].depth = depth;
                nodes[tail].id = id;
                nodes[tail].alpha = alpha;
                nodes[tail].color = color;
            }
            return;
        }

        /**
           If we are at the end, *and* we can impact the scene more
           meaningfully, then we replace the one at the end
           TODO: might lead to aliasing
        */
        if (nodes[nodeIdx].next == -1)
        {
            if (alpha - nodes[nodeIdx].alpha > IMPACT_ALPHA)
            {
                nodes[nodeIdx].depth = depth;
                nodes[nodeIdx].id = id;
                nodes[nodeIdx].alpha = alpha;
                nodes[nodeIdx].color = color;
            }
            return;
        }
        int newTail = nodes[tail].prev;
        nodes[tail].depth = depth;
        nodes[tail].id = id;
        nodes[tail].alpha = alpha;
        nodes[tail].color = color;
        if (nodes[nodeIdx].prev != -1)
        {
            nodes[nodes[nodeIdx].prev].next = tail;
            nodes[tail].prev = nodes[nodeIdx].prev;
        }
        else
        {
            // There is no prev for the inserted. That means we
            // become the new head, automatically.
            nodes[tail].prev = -1;
            head = tail;
        }
        nodes[tail].next = nodeIdx;
        nodes[nodeIdx].prev = tail;
        tail = newTail;
        nodes[newTail].next = -1;
    }

    void initAdaptiveFHost(MiniNode *nodes, size_t numNodes, int &head, int &tail)
    {
        initAdaptiveF(nodes, numNodes, head, tail);
    }

    void insertAdaptiveFHost(
        MiniNode *nodes, size_t numNodes,
        float depth, uint32_t id, float alpha, const float *color,
        int &head, int &tail)
    {
        insertAdaptiveF(
            nodes, numNodes, depth, id, alpha, reinterpret_cast<const glm::vec3 &>(*color),
            head, tail);
    }

    __device__ float obtainAlpha(
        const glm::uvec2 &pix, int width, int height,
        const gs::MathematicalEllipse &ellipse, const glm::vec4 &conicOpacity,
        const glm::vec2 &screenSpace, const ForwardParams &forwardParams)
    {
        float alpha = 0.0f;
        if (forwardParams.ellipseApprox)
        {
            // 1. We don't need to evaluate delta, not
            // actually. Transform pix to NDC.
            glm::vec2 pixNDC = glm::vec2((float) pix.x / width, (float) pix.y / height);
            pixNDC = pixNDC * 2.0f - 1.0f;
            float aspect = (float) width / height;
            pixNDC.x *= aspect;

            // 2. Plug it into our ellipse formula.
            // float ellipseVal = -glm::dot(pixNDC, collectedEllipses[j].A * pixNDC) +
            //  glm::dot(pixNDC, collectedEllipses[j].b) + collectedEllipses[j].c;

            float ellipseVal = (ellipse.A[0][0] * pixNDC.x + ellipse.A[1][0] * pixNDC.y) * pixNDC.x +
                (ellipse.A[0][1] * pixNDC.x + ellipse.A[1][1] * pixNDC.y) * pixNDC.y +
                ellipse.b[0] * pixNDC.x + ellipse.b[1] * pixNDC.y + ellipse.c;
            ellipseVal = -ellipseVal;

            if (ellipseVal > 0.0f)
            {
                return alpha;
            }
            alpha = 0.5f * conicOpacity.w;
        }
        else
        {
            glm::vec2 delta = screenSpace - glm::vec2(pix);

            if (!forwardParams.cosineApprox)
            {
                // "The alpha decays exponentially from the Gaussian center." I've read that somewhere in the paper. May not be
                // completely the same though. Well, it was either in paper or in code.
                // TODO: I have no idea what this equation means. SIBR explanation:
                // Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
                float power = -0.5f * (conicOpacity.x * delta.x * delta.x + conicOpacity.z * delta.y * delta.y) - conicOpacity.y * delta.x * delta.y;
                if (power > 0.0f)
                {
                    return alpha; // ?????
                }

                // Oh wait, here it is.
                // Eq. (2) from 3D Gaussian splatting paper.
                // Obtain alpha by multiplying with Gaussian opacity
                // and its exponential falloff from mean.
                // Avoid numerical instabilities (see paper appendix).
                alpha = exp(power);
            }
            else
            {
                float coeff = conicOpacity.x * delta.x * delta.x + conicOpacity.z * delta.y * delta.y + 2.0f * conicOpacity.y * delta.x * delta.y;
                if (coeff < 0.0f)
                {
                    return alpha;
                }
                alpha = approxNorm2(coeff);
            }
            alpha = glm::min(0.99f, conicOpacity.w * alpha);
        }

        return alpha;
    }

    /**
     * actually render stuffs to the screen.
     */
    __global__ void renderCUDA(const glm::uvec2 * __restrict__ ranges,
                               const uint32_t * __restrict__ pointList,
                               int width, int height,
                               const glm::vec2 * __restrict__ means2D,
                               const glm::vec3 * __restrict__ features, // "colors"
                               const glm::vec4 * __restrict__ conicOpacities,
                               const gs::MathematicalEllipse * __restrict__ ellipses,
                               const float * __restrict__ depths,
                               float * __restrict__ finalT, // "accumAlpha"
                               uint32_t * __restrict__ nContrib,
                               const glm::vec3 * __restrict__ background,
                               float * __restrict__ outColor,
                               ForwardParams forwardParams)
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
        __shared__ gs::MathematicalEllipse collectedEllipses[blockSize];
        __shared__ glm::vec3 collectedColors[blockSize];

        /**
           The adaptive OIT visibility function.
        */
        __shared__ float collectedDepths[blockSize];
        MiniNode adaptiveF[ADAPTIVE_FUNC_SIZE];
        int adaptiveFHead, adaptiveFTail;
        initAdaptiveF(adaptiveF, ADAPTIVE_FUNC_SIZE, adaptiveFHead, adaptiveFTail);

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
                collectedEllipses[block.thread_rank()] = ellipses[collId];
                collectedDepths[block.thread_rank()] = depths[collId];
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

                const gs::MathematicalEllipse &ellipse = collectedEllipses[j];
                const glm::vec4 &conicOpacity = collectedConicOpacities[j];
                const glm::vec2 &screenSpace = collectedXYs[j];
                float alpha = obtainAlpha(pix, width, height, ellipse, conicOpacity,
                                          screenSpace, forwardParams);

                if (alpha < 1.0f / 255.0f)
                {
                    continue;
                }

                if (forwardParams.debugCosineApprox)
                {
                    // collectedColors[j] = glm::vec3(delta.x, delta.y, fabs(delta.x * delta.x));
                }

                if (forwardParams.adaptiveOIT)
                {
                    // Use premultiplied colors
                    insertAdaptiveF(
                        adaptiveF, ADAPTIVE_FUNC_SIZE,
                        collectedDepths[j], collectedIds[j],
                        alpha, alpha * collectedColors[j],
                        adaptiveFHead, adaptiveFTail);
                }
                else
                {
                    // Test the remaining alpha and see if it makes sense to continue the blend.
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
                }

                lastContributor = contributor;
            }
        }

        // It is done. Write all things to output buffer
        if (inside)
        {
            if (forwardParams.adaptiveOIT)
            {
                /**
                   If we are using adaptive OIT, then so far we have
                   not calculated anything whatsoever; we are only
                   starting the blend now.
                */
                int it = adaptiveFHead;
                float visibility = 1.0f;
                while (it != -1 && adaptiveF[it].id != -1)
                {
                    color += visibility * adaptiveF[it].color;
                    visibility *= (1.0f - adaptiveF[it].alpha);
                    it = adaptiveF[it].next;
                }
            }

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
                const gs::MathematicalEllipse *ellipses,
                const float *depths,
                float *finalT, // "accumAlpha"
                uint32_t *nContrib,
                const glm::vec3 *background,
                float *outColor,
                ForwardParams forwardParams)
    {
        renderCUDA<<<grid, block>>>(ranges, pointList, width, height,
                                    means2D, colors, conicOpacities, ellipses, depths,
                                    finalT, nContrib, background, outColor, forwardParams);
    }

    void forward(std::function<char *(size_t)> geometryBuffer,
                 std::function<char *(size_t)> binningBuffer,
                 std::function<char *(size_t)> imageBuffer,
                 std::function<char *(size_t)> compositeLayerBuffer,
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
                 const float *perspectiveMatrix, // CUDA mat4
                 const float *projMatrix, // CUDA mat4: perspective * view
                 const float *camPos, // CUDA vec3
                 float tanFOVx, float tanFOVy, // for focal length calculation
                 bool prefiltered, // Unused
                 float *outColor, // The outColor array
                 int *radii, // Unused
                 int *rects, // CUDA rects for fast culling
                 float *boxMin, // Unused; bounding box I think
                 float *boxMax,
                 ForwardParams forwardParams)
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
        char *imBuffer = imageBuffer(imageBufferSize + 228);
        ImageState imState = ImageState::fromChunk(imBuffer, width * height);

        // Unlike imBuffer, which records per-pixel information,
        // composite buffer acts as a separate pass, and logs on
        // graphical debug information.
        size_t compositeBufferSize = width * height * 3 * sizeof(float);
        char *compositeBufferRaw = compositeLayerBuffer(compositeBufferSize + 128);
        cudaMemset(compositeBufferRaw, 0, compositeBufferSize);
        float *compositeBuffer = reinterpret_cast<float *>(compositeBufferRaw);

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
                   viewMatrix, perspectiveMatrix, projMatrix,
                   (glm::vec3 *) camPos,
                   width, height,
                   focalDist,
                   tanFOVx, tanFOVy,
                   radii,
                   geomState.means2D,
                   geomState.depths,
                   geomState.cov3D,
                   geomState.rgb,
                   geomState.conicOpacity,
                   geomState.ellipsoids,
                   geomState.ellipses,
                   tileGrid,
                   geomState.tilesTouched,
                   prefiltered,
                   (glm::ivec2 *) rects,
                   minn, maxx,
                   compositeBuffer,
                   forwardParams);

        // Preprocessed; next up get the full list of touched tiles
        cub::DeviceScan::InclusiveSum(geomState.scanningSpace, geomState.scanSize, geomState.tilesTouched, geomState.pointOffsets, numGaussians);
        cudaMemcpy(&geomState.numRendered, &geomState.pointOffsets[numGaussians - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Pointless to go any farther
        if (geomState.numRendered == 0)
        {
            cudaMemset(outColor, 0, compositeBufferSize);
            composite(width, height, outColor, compositeBuffer);
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
                                                               radii, tileGrid, (glm::ivec2 *) rects, forwardParams);

        int highestBit = getHigherMsb(tileGrid.x * tileGrid.y);

        // Sort Gaussians
        cub::DeviceRadixSort::SortPairs(binningState.listSortingSpace, binningState.sortingSize,
                                        binningState.pointListKeysUnsorted, binningState.pointListKeys,
                                        binningState.pointListUnsorted, binningState.pointList,
                                        geomState.numRendered, 0, highestBit + 32);

        // Determine tile ranges
        cudaMemset(imState.ranges, 0, sizeof(glm::uvec2) * width * height);
        identifyTileRanges<<<(geomState.numRendered + 255) / 256, 256>>>(geomState.numRendered, binningState.pointListKeys,
                                                                         imState.ranges, forwardParams);

        const glm::vec3 *colorsPtr = colorsPrecomp ? (const glm::vec3 *) colorsPrecomp : geomState.rgb;
        render(tileGrid, block,
               imState.ranges, binningState.pointList,
               width, height,
               geomState.means2D, colorsPtr, geomState.conicOpacity,
               geomState.ellipses, geomState.depths,
               imState.accumAlpha, imState.nContrib,
               (const glm::vec3 *) background,
               outColor, forwardParams);

        if (forwardParams.highlightBlockX != -1 && forwardParams.highlightBlockY != -1)
        {
            blitRectHost(compositeBuffer, width, height,
                         forwardParams.highlightBlockX * BLOCK_W,
                         forwardParams.highlightBlockY * BLOCK_H,
                         BLOCK_W, BLOCK_H,
                         glm::vec3(1.0f, 0.5f, 0.0f));
        }

        composite(width, height, outColor, compositeBuffer);
    }
};
