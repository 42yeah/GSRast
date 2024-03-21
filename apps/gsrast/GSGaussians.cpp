#include "GSGaussians.hpp"
#include "CameraBase.hpp"
#include "Config.hpp"
#include "FirstPersonCamera.hpp"
#include "GSPointCloud.hpp"
#include "CudaBuffer.hpp"
#include "Framebuffer.hpp"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <glm/matrix.hpp>
#include <memory>
#include <vector>
#include <glm/gtc/type_ptr.hpp>
#include <GSCuda.cuh>

#define USE_GSCUDA

#ifdef USE_GSCUDA 
#define FORWARD gscuda::forward
#else 
#include <rasterizer.h>
#define FORWARD CudaRasterizer::Rasterizer::forward
#endif 

// Functionals which acts as little classes (buffer holders.)
// Taken directly from SIBR_Viewers.
std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S) {
    auto lambda = [ptr, &S](size_t N)
    {
        if (N > S)
        {
            if (*ptr)
            {
                CHECK_CUDA_ERROR(cudaFree(*ptr));
            }
            CHECK_CUDA_ERROR(cudaMalloc(ptr, 2 * N));
            S = 2 * N;
        }
        return reinterpret_cast<char *>(*ptr);
    };
    return lambda;
}

GSGaussians::GSGaussians(int width, int height, FirstPersonCamera::Ptr camera) : GSPointCloud()
{
    _width = width;
    _height = height;
    _numGaussians = 0;
    _interopTex = std::make_shared<InteropBuffer>();
    _interopTex->allocate(_width * _height * sizeof(float) * NUM_CHANNELS);

    _positions = std::make_shared<CudaBuffer<float> >();
    _rotations = std::make_shared<CudaBuffer<float> >();
    _scales = std::make_shared<CudaBuffer<float> >();
    _opacities = std::make_shared<CudaBuffer<float> >();
    _shs = std::make_shared<CudaBuffer<float> >();
    _view = std::make_shared<CudaBuffer<glm::mat4> >();
    _perspective = std::make_shared<CudaBuffer<glm::mat4> >();
    _projection = std::make_shared<CudaBuffer<glm::mat4> >();
    _camPos = std::make_shared<CudaBuffer<glm::vec3> >();
    _background = std::make_shared<CudaBuffer<glm::vec3> >();
    _rects = std::make_shared<CudaBuffer<int> >();

    _view->allocate(sizeof(glm::mat4));
    _projection->allocate(sizeof(glm::mat4));
    _perspective->allocate(sizeof(glm::mat4));
    _camPos->allocate(sizeof(glm::vec3));
    _background->allocate(sizeof(glm::vec3));

    _camera = camera;

    _geomPtr = nullptr;
    _binningPtr = nullptr;
    _imgPtr = nullptr;
    _compositePtr = nullptr;
    _allocatedGeom = 0;
    _allocatedBinning = 0;
    _allocatedImg = 0;
    _allocatedComposite = 0;
    _geomBufferFunc = resizeFunctional(&_geomPtr, _allocatedGeom);
    _binningBufferFunc = resizeFunctional(&_binningPtr, _allocatedBinning);
    _imgBufferFunc = resizeFunctional(&_imgPtr, _allocatedImg);
    _compositeLayerBufferFunc = resizeFunctional(&_compositePtr, _allocatedComposite);

    _splatData = nullptr;
    _forwardParams.cosineApprox = false;
    _forwardParams.debugCosineApprox = false;
    _forwardParams.ellipseApprox = false;
    _forwardParams.adaptiveOIT = false;
    _forwardParams.ellipseApproxFocalDist = 2.404f;
    _forwardParams.selected = -1;
    _forwardParams.highlightBlockX = -1;
    _forwardParams.highlightBlockY = -1;
}

GSGaussians::~GSGaussians()
{
    if (_geomPtr)
    {
        CHECK_CUDA_ERROR(cudaFree(_geomPtr));
    }
    if (_binningPtr)
    {
        CHECK_CUDA_ERROR(cudaFree(_binningPtr));
    }
    if (_imgPtr)
    {
        CHECK_CUDA_ERROR(cudaFree(_imgPtr));
    }
    if (_compositePtr)
    {
	CHECK_CUDA_ERROR(cudaFree(_compositePtr));
    }
}

int GSGaussians::getWidth() const
{
    return _width;
}

int GSGaussians::getHeight() const
{
    return _height;
}

bool GSGaussians::configureFromSplatData(const SplatData::Ptr &splatData, const ShaderBase::Ptr &shader)
{
    if (!splatData->isValid())
    {
        return false;
    }

    _splatData = splatData;
    _center = splatData->getCenter();
    _numGaussians = splatData->getNumGaussians();
    _bbox = splatData->getBBox();

    _positions->memcpy((float *) splatData->getPositions().data(),
                       (int) splatData->getPositions().size() * sizeof(glm::vec4));

    _scales->memcpy((float *) splatData->getScales().data(),
                    (int) splatData->getScales().size() * sizeof(glm::vec4));

    _shs->memcpy((float *) splatData->getSHs().data(),
                 (int) splatData->getSHs().size() * sizeof(SHs<3>));

    _rotations->memcpy((float *) splatData->getRotations().data(),
                       (int) splatData->getRotations().size() * sizeof(glm::vec4));

    _opacities->memcpy((float *) splatData->getOpacities().data(),
                       (int) splatData->getOpacities().size() * sizeof(float));

    // ???
    _rects->allocate(sizeof(int) * 2 * _numGaussians);

    std::cout << "Loading report: " << std::endl
        << "positions: " << _positions->size() << std::endl
        << "scales: " << _scales->size() << std::endl
        << "SHs: " << _shs->size() << std::endl
        << "rotations: " << _rotations->size() << std::endl
        << "opacities: " << _opacities->size() << std::endl;

    _shader = shader;

    _background->set(glm::vec3(0.0f, 0.0f, 0.0f));

    configure(Framebuffer::rectData, 6, sizeof(Framebuffer::rectData), _shader);

    return true;
}

void GSGaussians::draw()
{
    glm::mat4 view = _camera->getView();
    glm::mat4 projection = _camera->getPerspective() * _camera->getView();

    view = glm::transpose(view);
    projection = glm::transpose(projection);

    // Once we're transposed, we will be operating on rows
    // view[1] *= -1.0f;
    view[2] *= -1.0f;
    // projection[1] *= -1.0f;

    view = glm::transpose(view);
    projection = glm::transpose(projection);

    _view->set(view);
    _perspective->set(_camera->getPerspective());
    _projection->set(projection);
    _camPos->set(_camera->getPosition());

    float tanFOVy = tan(_camera->getFOV() * 0.5f);
    float tanFOVx = tanFOVy * ((float) _width / _height);

    CHECK_CUDA_ERROR(
        FORWARD(_geomBufferFunc,
                _binningBufferFunc,
                _imgBufferFunc,
		_compositeLayerBufferFunc,
                _numGaussians,
                3,
                16,
                (const float *) _background->getPtr(),
                _width,
                _height,
                _positions->getPtr(),
                _shs->getPtr(),
                nullptr,
                _opacities->getPtr(),
                _scales->getPtr(),
                1.0f,
                _rotations->getPtr(),
                nullptr,
                (const float *) _view->getPtr(),
		(const float *) _perspective->getPtr(),
                (const float *) _projection->getPtr(),
                (const float *) _camPos->getPtr(),
                tanFOVx,
                tanFOVy,
                false,
                _interopTex->getPtr(),
                nullptr,
                _rects->getPtr(),
                nullptr,
                nullptr,
                _forwardParams)
    );

    // Now directly render-copy the result by treating the interopTex as an SSBO
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _interopTex->getBuffer());
    BufferGeo::draw();
}

gscuda::gs::GeometryState GSGaussians::mapGeometryState() const
{
    char *chunk = (char *) _geomPtr;
    gscuda::gs::GeometryState ret = gscuda::gs::GeometryState::fromChunk(chunk, _numGaussians);
    return ret;
}

const gscuda::ForwardParams &GSGaussians::getForwardParams() const
{
    return _forwardParams;
}

void GSGaussians::setForwardParams(const gscuda::ForwardParams &forwardParams)
{
    _forwardParams = forwardParams;
}
