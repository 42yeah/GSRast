#include "GSRastWindow.hpp"
#include "BufferGeo.hpp"
#include "DrawBase.hpp"
#include "Window.hpp"
#include "Config.hpp"
#include "GSPointCloud.hpp"
#include "SplatShader.hpp"
#include "CudaBuffer.hpp"
#include "GSGaussians.hpp"
#include "Inspector.hpp"
#include "apps/gsrast/GSEllipsoids.hpp"
#include <GLFW/glfw3.h>
#include <memory>
#include <iostream>


GSRastWindow::GSRastWindow() : Window(WINDOW_TITLE, DEFAULT_WINDOW_W, DEFAULT_WINDOW_H)
{
    configureFirstPersonCamera();
    _firstPersonCamera->setInvertUp(true);
    _pcShader = std::make_shared<PointCloudShader>(_firstPersonCamera);
    _splatShader = std::make_shared<SplatShader>(_firstPersonCamera);
    _copyShader = std::make_shared<CopyShader>();
    _splatData = std::make_shared<SplatData>("data.ply");
    _renderSelector = std::make_shared<RenderSelector>();
    _renderSelector->reset(nullptr);
    _visMode = VisMode::PointCloud;

    // setup initial camera pose.
    BBox bbox = _splatData->getBBox();
    glm::vec3 span = bbox.span();
    float far = glm::max(glm::max(span.x, span.y), span.z);

    _firstPersonCamera->setPosition(glm::vec3(0.0f) - glm::vec3(0.0f, 0.0f, 5.0f));
    _firstPersonCamera->lookAt(glm::vec3(0.0f));
    _firstPersonCamera->setNearFar(0.001f * far, far);
    _firstPersonCamera->setSpeed(far * 0.1f);

    pointCloudMode();
    _samplerShader = std::make_shared<SamplerShader>();
    _framebuffer = std::make_shared<Framebuffer>(getWidth(), getHeight(), false, _samplerShader);
    _framebuffer->addDrawable(_renderSelector);
    _framebuffer->clear(glm::vec4(0.1f, 0.1f, 0.1f, 1.0f));
    addDrawable(_framebuffer);

    _inspector = std::make_shared<Inspector>(this);
    addDrawable(_inspector);

    glPointSize(2.0f);
}

GSRastWindow::~GSRastWindow()
{

}

void GSRastWindow::keyCallback(int key, int scancode, int action, int mods)
{
    Window::keyCallback(key, scancode, action, mods);

    if (key == GLFW_KEY_R && action == GLFW_PRESS)
    {
        std::cout << "OpenGL error: " << glGetError() << std::endl;
    }
    if (key == GLFW_KEY_DOWN && action == GLFW_PRESS)
    {
        _firstPersonCamera->setSpeed(_firstPersonCamera->getSpeed() * 0.5f);
    }
    if (key == GLFW_KEY_UP && action == GLFW_PRESS)
    {
        _firstPersonCamera->setSpeed(_firstPersonCamera->getSpeed() * 2.0f);
    }
}

const SplatData::Ptr &GSRastWindow::getSplatData() const
{
    return _splatData;
}

const PointCloudShader::Ptr &GSRastWindow::getPCShader() const
{
    return _pcShader;
}

const SplatShader::Ptr &GSRastWindow::getSplatShader() const
{
    return _splatShader;
}

const CopyShader::Ptr &GSRastWindow::getCopyShader() const
{
    return _copyShader;
}

void GSRastWindow::pointCloudMode()
{
    // Draw point clouds (GSPC)
    GSPointCloud::Ptr gsPtr = std::make_shared<GSPointCloud>();
    _renderSelector->reset(nullptr);
    if (!gsPtr->configureFromSplatData(_splatData, _pcShader))
    {
        std::cerr << "Could not load PC from PLY?" << std::endl;
    }
    else
    {
        _firstPersonCamera->setInvertUp(true);
        _visMode = VisMode::PointCloud;
        _renderSelector->reset(gsPtr);
    }
}

void GSRastWindow::ellipsoidsMode()
{
    // Draw ellipsoids
    GSEllipsoids::Ptr gsPtr = std::make_shared<GSEllipsoids>();
    if (!gsPtr->configureFromSplatData(_splatData, _splatShader))
    {
        std::cerr << "Could not load PC from PLY?" << std::endl;
    }
    else
    {
        _firstPersonCamera->setInvertUp(true);
        _visMode = VisMode::Ellipsoids;
        _renderSelector->reset(gsPtr);
    }
}

void GSRastWindow::gaussianMode()
{
    // Draw Gaussians
    GSGaussians::Ptr gsPtr = std::make_shared<GSGaussians>(_w, _h, _firstPersonCamera);
    if (!gsPtr->configureFromSplatData(_splatData, _copyShader))
    {
        std::cerr << "Cannot configure Gaussians?" << std::endl;
    }
    else
    {
        _firstPersonCamera->setInvertUp(true);
        _visMode = VisMode::Gaussians;
        _renderSelector->reset(gsPtr);
    }
}

void GSRastWindow::drawFrame()
{
    _framebuffer->drawFrame();
    Window::drawFrame();
}

VisMode GSRastWindow::getVisMode() const
{
    return _visMode;
}

const Framebuffer::Ptr &GSRastWindow::getFramebuffer() const
{
    return _framebuffer;
}

const RenderSelector::Ptr &GSRastWindow::getRenderSelector() const
{
    return _renderSelector;
}
