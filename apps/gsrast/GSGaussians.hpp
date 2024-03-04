#pragma once

#include "Config.hpp"
#include "FirstPersonCamera.hpp"
#include "GSPointCloud.hpp"
#include <functional>
#include "ShaderBase.hpp"
#include "CudaBuffer.hpp"
#include "SplatData.hpp"

/**
 * This part of application invokes the graphdeco-inria/diff-gaussian-rasterization to render Gaussian Splats.
 * Kerbl, Bernhard and Kopanas, Georgios and Leimkuhler, Thomas and Drettakis, George:
 * 3D Gaussian Splatting for Real-Time Radiance Field Rendering. https://github.com/graphdeco-inria/diff-gaussian-rasterization
 */
class GSGaussians : public GSPointCloud
{
public:
    CLASS_PTRS(GSGaussians)

    GSGaussians(int width, int height, FirstPersonCamera::Ptr camera);
    virtual ~GSGaussians();

    int getWidth() const;
    int getHeight() const;

    virtual bool configureFromSplatData(const SplatData::Ptr &splatData, const ShaderBase::Ptr &shader) override;
    virtual void draw() override;

protected:
    int _width, _height, _numGaussians;

    CudaBuffer<float>::Ptr _positions, _rotations, _shs, _scales, _opacities;
    CudaBuffer<glm::mat4>::Ptr _view, _projection;
    CudaBuffer<glm::vec3>::Ptr _camPos, _background;
    CudaBuffer<int>::Ptr _rects;

    InteropBuffer::Ptr _interopTex;

    ShaderBase::Ptr _shader;
    FirstPersonCamera::Ptr _camera;

    std::function<char *(size_t)> _geomBufferFunc, _binningBufferFunc, _imgBufferFunc;
    void *_geomPtr, *_binningPtr, *_imgPtr;
    size_t _allocatedGeom, _allocatedBinning, _allocatedImg;
    SplatData::Ptr _splatData;
};
