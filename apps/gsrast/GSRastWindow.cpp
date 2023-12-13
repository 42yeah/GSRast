#include "GSRastWindow.hpp"
#include "Window.hpp"
#include "Config.hpp"
#include "GSPointCloud.hpp"
#include "apps/gsrast/GSEllipsoids.hpp"
#include "apps/gsrast/SplatShader.hpp"
#include <GLFW/glfw3.h>
#include <memory>
#include <iostream>


GSRastWindow::GSRastWindow() : Window(WINDOW_TITLE, DEFAULT_WINDOW_W, DEFAULT_WINDOW_H)
{
    configureFirstPersonCamera();
    _pcShader = std::make_shared<PointCloudShader>(_firstPersonCamera);
    _splatShader = std::make_shared<SplatShader>(_firstPersonCamera);

    // Draw point clouds (GSPC)
    // GSPointCloud::Ptr gsPtr = std::make_shared<GSPointCloud>();
    // if (!gsPtr->configureFromPly("data.ply", _pcShader))
    // {
    //     std::cerr << "Could not load PC from PLY?" << std::endl;
    // }
    // else
    // {
    //     addDrawable(gsPtr);
    //     _firstPersonCamera->setPosition(gsPtr->getBBox().center - glm::vec3(0.0f, 0.0f, 5.0f));
    //     _firstPersonCamera->lookAt(gsPtr->getBBox().center);
    // }

    // Draw ellipsoids
    GSEllipsoids::Ptr gsPtr = std::make_shared<GSEllipsoids>();
    if (!gsPtr->configureFromPly("data.ply", _splatShader))
    {
        std::cerr << "Could not load PC from PLY?" << std::endl;
    }
    else
    {
        BBox bbox = gsPtr->getBBox();
        glm::vec3 span = bbox.span();
        float far = glm::max(glm::max(span.x, span.y), span.z);

        addDrawable(gsPtr);
        _firstPersonCamera->setPosition(gsPtr->getCenter() - glm::vec3(0.0f, 0.0f, 5.0f));
        _firstPersonCamera->lookAt(gsPtr->getCenter());
        _firstPersonCamera->setNearFar(0.001f * far, far);
    }

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
}
