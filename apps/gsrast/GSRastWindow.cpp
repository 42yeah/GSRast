#include "GSRastWindow.hpp"
#include "Window.hpp"
#include "Config.hpp"
#include "GSPointCloud.hpp"
#include <memory>
#include <iostream>

GSRastWindow::GSRastWindow() : Window(WINDOW_TITLE, DEFAULT_WINDOW_W, DEFAULT_WINDOW_H)
{
    configureFirstPersonCamera();
    _orbitalShader = std::make_shared<OrbitalShader>(getCamera());

    GSPointCloud::Ptr gsPtr = std::make_shared<GSPointCloud>();
    if (!gsPtr->configureFromPly("data.ply", _orbitalShader))
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
