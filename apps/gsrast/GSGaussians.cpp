#include "GSGaussians.hpp"
#include "CameraBase.hpp"
#include "Config.hpp"
#include "FirstPersonCamera.hpp"
#include "GSPointCloud.hpp"
#include "CudaBuffer.hpp"
#include "config.h"
#include "Framebuffer.hpp"
#include <cuda_runtime_api.h>
#include <glm/matrix.hpp>
#include <rasterizer.h>
#include <memory>
#include <vector>
#include <glm/gtc/type_ptr.hpp>

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
    _projection = std::make_shared<CudaBuffer<glm::mat4> >();
    _camPos = std::make_shared<CudaBuffer<glm::vec3> >();
    _background = std::make_shared<CudaBuffer<glm::vec3> >();
    _rects = std::make_shared<CudaBuffer<int> >();

    _view->allocate(sizeof(glm::mat4));
    _projection->allocate(sizeof(glm::mat4));
    _camPos->allocate(sizeof(glm::vec3));
    _background->allocate(sizeof(glm::vec3));

    _camera = camera;

    _geomPtr = nullptr;
    _binningPtr = nullptr;
    _imgPtr = nullptr;
    _allocatedGeom = 0;
    _allocatedBinning = 0;
    _allocatedImg = 0;
    _geomBufferFunc = resizeFunctional(&_geomPtr, _allocatedGeom);
    _binningBufferFunc = resizeFunctional(&_binningPtr, _allocatedBinning);
    _imgBufferFunc = resizeFunctional(&_imgPtr, _allocatedImg);
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
}

int GSGaussians::getWidth() const
{
    return _width;
}

int GSGaussians::getHeight() const
{
    return _height;
}

bool GSGaussians::configureFromPly(const std::string &path, ShaderBase::Ptr shader)
{
    const auto splats = loadFromSplatsPly(path);
    if (!splats->valid)
    {
        return false;
    }

    _center = glm::vec3(0.0f);
    _numGaussians = splats->numSplats;

    std::vector<glm::vec3> points;
    points.resize(splats->numSplats);
    _bbox.reset();
    for (int i = 0; i < splats->numSplats; i++)
    {
        points[i] = splats->splats[i].position;
        _center += points[i];
        _bbox.enclose(points[i]);
    }
    if (splats->numSplats > 0)
    {
        _center /= points.size();
    }
    _positions->memcpy((float *) points.data(), (int) points.size() * sizeof(glm::vec3));

    for (int i = 0; i < splats->numSplats; i++)
    {
        const glm::vec3 &scale = splats->splats[i].scale;
        points[i] = glm::vec3(exp(scale.x), exp(scale.y), exp(scale.z));
    }
    _scales->memcpy((float *) points.data(), (int) points.size() * sizeof(glm::vec3));
    points = std::vector<glm::vec3>(); // Clear the vector data

    std::vector<SHs<3> > shs;
    shs.resize(splats->numSplats);
    for (int i = 0; i < splats->numSplats; i++)
    {
        shs[i] = splats->splats[i].shs;
    }
    _shs->memcpy((float *) shs.data(), (int) shs.size() * sizeof(SHs<3>));
    shs = std::vector<SHs<3> >();

    std::vector<glm::vec4> rotations;
    rotations.resize(splats->numSplats);
    for (int i = 0; i < splats->numSplats; i++)
    {
        rotations[i] = glm::normalize(splats->splats[i].rotation);
    }
    _rotations->memcpy((float *) rotations.data(), (int) rotations.size() * sizeof(glm::vec4));
    rotations = std::vector<glm::vec4>();

    std::vector<float> opacities;
    opacities.resize(splats->numSplats);
    for (int i = 0; i < splats->numSplats; i++)
    {
        opacities[i] = sigmoid(splats->splats[i].opacity);
    }
    _opacities->memcpy((float *) opacities.data(), (int) opacities.size() * sizeof(float));
    opacities = std::vector<float>();

    // ???
    _rects->allocate(sizeof(int) * 2 * splats->numSplats);

    std::cout << "Loading report: " << std::endl
        << "positions: " << _positions->size() << std::endl
        << "scales: " << _scales->size() << std::endl
        << "SHs: " << _shs->size() << std::endl
        << "rotations: " << _rotations->size() << std::endl
        << "opacities: " << _opacities->size() << std::endl;

    _shader = shader;

    _background->set(glm::vec3(1.0f, 0.0f, 1.0f));

    configure(Framebuffer::rectData, 6, sizeof(Framebuffer::rectData), _shader);

    return true;
}

void GSGaussians::draw()
{
    glm::mat4 view = _camera->getView();
    glm::mat4 projection = _camera->getPerspective() * _camera->getView();

    // Once we're transposed, we will be operating on rows
    view[1] *= -1.0f;
    view[2] *= -1.0f;
    projection[1] *= -1.0f;

    _view->set(view);
    _projection->set(projection);
    _camPos->set(_camera->getPosition());

    float tanFOVy = tan(_camera->getFOV() * 0.5f);
    float tanFOVx = tanFOVy * ((float) _width / _height);

    CHECK_CUDA_ERROR(
        CudaRasterizer::Rasterizer::forward(_geomBufferFunc,
                                            _binningBufferFunc,
                                            _imgBufferFunc,
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
                                            (const float *) _projection->getPtr(),
                                            (const float *) _camPos->getPtr(),
                                            tanFOVx,
                                            tanFOVy,
                                            false,
                                            _interopTex->getPtr(),
                                            nullptr,
                                            _rects->getPtr(),
                                            nullptr,
                                            nullptr)
    );

    // Now directly render-copy the result by treating the interopTex as an SSBO
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _interopTex->getBuffer());
    BufferGeo::draw();
}

