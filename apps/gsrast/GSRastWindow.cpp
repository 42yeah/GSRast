#include "GSRastWindow.hpp"
#include "Window.hpp"
#include "Config.hpp"
#include "GSPointCloud.hpp"
#include "SplatShader.hpp"
#include "CudaBuffer.hpp"
#include "GSGaussians.hpp"
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

    // Draw point clouds (GSPC)
    // GSPointCloud::Ptr gsPtr = std::make_shared<GSPointCloud>();
    // if (!gsPtr->configureFromSplatData(_splatData, _pcShader))
    // {
    //     std::cerr << "Could not load PC from PLY?" << std::endl;
    // }
    // else
    // {
    //     addDrawable(gsPtr);
    //     _firstPersonCamera->setPosition(gsPtr->getCenter() - glm::vec3(0.0f, 0.0f, 5.0f));
    //     _firstPersonCamera->lookAt(gsPtr->getCenter());
    // }

    // Draw ellipsoids
    // GSEllipsoids::Ptr gsPtr = std::make_shared<GSEllipsoids>();
    // if (!gsPtr->configureFromSplatData(_splatData, _splatShader))
    // {
    //     std::cerr << "Could not load PC from PLY?" << std::endl;
    // }
    // else
    // {
    //     BBox bbox = gsPtr->getBBox();
    //     glm::vec3 span = bbox.span();
    //     float far = glm::max(glm::max(span.x, span.y), span.z);
    //
    //     addDrawable(gsPtr);
    //     _firstPersonCamera->setPosition(gsPtr->getCenter() - glm::vec3(0.0f, 0.0f, 5.0f));
    //     _firstPersonCamera->lookAt(gsPtr->getCenter());
    //     _firstPersonCamera->setNearFar(0.001f * far, far);
    //     _firstPersonCamera->setSpeed(far * 0.1f);
    // }

    // Draw Gaussians
    GSGaussians::Ptr gsPtr = std::make_shared<GSGaussians>(_w, _h, _firstPersonCamera);
    if (!gsPtr->configureFromSplatData(_splatData, _copyShader))
    {
        std::cerr << "Cannot configure Gaussians?" << std::endl;
    }
    else
    {
        _firstPersonCamera->setPosition(gsPtr->getCenter() - glm::vec3(0.0f, 0.0f, 5.0f));
        _firstPersonCamera->lookAt(gsPtr->getCenter());
        BBox bbox = gsPtr->getBBox();
        glm::vec3 span = bbox.span();
        float far = glm::max(glm::max(span.x, span.y), span.z);
        _firstPersonCamera->setNearFar(0.001f * far, far);
        _firstPersonCamera->setSpeed(far * 0.1f);
    }
    addDrawable(gsPtr);
    // gsPtr->draw();

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
