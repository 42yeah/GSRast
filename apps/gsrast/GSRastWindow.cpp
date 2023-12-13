#include "GSRastWindow.hpp"
#include "Window.hpp"
#include "Config.hpp"
#include "GSPointCloud.hpp"
#include "apps/gsrast/GSEllipsoids.hpp"
#include "apps/gsrast/SplatShader.hpp"
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
        addDrawable(gsPtr);
        _firstPersonCamera->setPosition(gsPtr->getBBox().center - glm::vec3(0.0f, 0.0f, 5.0f));
        _firstPersonCamera->lookAt(gsPtr->getBBox().center);
    }

    glPointSize(2.0f);
}

GSRastWindow::~GSRastWindow()
{

}
