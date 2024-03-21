#pragma once 

#include <cuda_runtime.h>
#include <functional>
#include <glm/glm.hpp>
#include "AuxBuffer.cuh"

#define CHECK_CUDA_ERROR(A) \
A; \
cudaDeviceSynchronize(); \
if (cudaPeekAtLastError() != cudaSuccess) \
std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(cudaGetLastError()) << "?" << std::endl;


/**
 * gscuda is the CUDA part of the GSRast, where we try to rasterize 
 * Gaussians. It is also a drop-in replacement of CudaRasterizer. 
 * The implementation of this function will strictly follow 
 * the implementation of diff_gaussian_rasterizer. 
 */
namespace gscuda 
{
    struct ForwardParams
    {
        bool cosineApprox;
        bool debugCosineApprox;
        bool ellipseApprox;
        bool adaptiveOIT;

	float ellipseApproxFocalDist; // Focal distance to render the
				      // approximated ellipses.
	int selected;
	int highlightBlockX, highlightBlockY;
    };

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

    /**
       Given *ONE* input axis (`axes[0]`), completes the rest
       and fill them into the matrix. Note, the input matrix MUST be
       incomplete. Or they WILL be overwritten!
    */
    __host__ __device__ void completeAxes(float *axes);

    /*
      Implementation of a memory-continuous linked list.
      Looks like this:

      [0 | depth | alpha decay | next]
      [1 | depth | alpha decay | next]
      ...
    */
    struct MiniNode
    {
	float depth;
	int id;
	int prev;
	int next;
    };

    struct KeyValue
    {
	int key;
	float value;
    };

    /**
       Construct an adaptive OIT function. See:
       https://www.intel.com/content/dam/develop/external/us/en/documents/37944-adaptive-transparency-hpg11.pdf
       In our case, we will build a visibility function in the form of
       a mini linked list in limited size (5), and sample it during
       the course of the rendering.

       The adaptiveF should be something like this:
       0.1 | [-]0.5, 0.2 | [-]0.3, ...
       And during the course of rendering, this will act as a CDF of
       sorts. WARNING: the linked list size MUST be greater than 1!
    */
    void constructAdaptiveFHost(const float *depths,
				const unsigned int *ids,
				const unsigned int *range,
				MiniNode *nodes, size_t numNodes,
				int &head, int &tail);

    /**
       The adaptive visibility function becomes a simple linked list
       traversal. I guess this can be mildly optimized. But I am not
       going for it.
    */
    int sampleAdaptiveFHost(MiniNode *nodes, size_t numNodes,
			    int head, float depth);

    void render(const dim3 grid, const dim3 block,
                const glm::uvec2 *ranges, const uint32_t *pointList,
                int width, int height,
                const glm::vec2 *means2D,
                const glm::vec3 *colors,
                const glm::vec4 *conicOpacities,
                float *finalT, // "accumAlpha"
                uint32_t *nContrib,
                const glm::vec3 *background,
                float *outColor,
                ForwardParams forwardParams);

    /**
     * I am the forward function responsible for
     * rendering stuffs onto the outColor array.
     * I am different from diff_gaussian_rasterizer's parameters
     * because I am too lazy to create another copy of the massive
     * positions and scales, and therefore they use glm::vec4.
     */
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
                 ForwardParams forwardParams);

    /**
       Given two color layers, one in the bottom and one on the top,
       my job is to composite both of them. Alpha is not part of the
       consideration; therefore, I am really only suitable for
       compositing debug data.
    */
    void composite(int width, int height, float *bottom, float *top);

    /**
       Host code for debugging.
       This can convert quaternions (vec4 as input) to rotation
       matrices (mat3 as output.)
       nvcc actually doens't produce the same name mangling as MSVC
       somehow; that's why we need to guarantee that all of them are
       primitive types.
    */
    void quatToMatHost(float *mat, const float *q);

    void ellipsoidFromGaussianHost(gs::MathematicalEllipsoid *ellip,
				   const float *rot, // mat3,
				   const float *scl, // vec4,
				   const float *center); // vec4

    void projectEllipsoidHost(gs::MathematicalEllipse *ellipse,
			      const gs::MathematicalEllipsoid *ellipsoid,
			      const float *camPos, // vec3
			      const float *planeAxes, // mat3
			      const float projectedDistance);
};
