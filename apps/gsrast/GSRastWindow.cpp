#include "GSRastWindow.hpp"
#include "BufferGeo.hpp"
#include "FirstPersonCamera.hpp"
#include "Window.hpp"
#include "Config.hpp"
#include <memory>
#include <iostream>

GSRastWindow::GSRastWindow() : Window(WINDOW_TITLE, DEFAULT_WINDOW_W, DEFAULT_WINDOW_H)
{
    configureFirstPersonCamera();
    _orbitalShader = std::make_shared<OrbitalShader>(getCamera());
    float tri[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };
    BufferGeo::Ptr bufGeo = std::make_shared<BufferGeo>();
    bufGeo->configure(tri, 3, sizeof(tri), _orbitalShader);

    addDrawable(bufGeo);
}

GSRastWindow::~GSRastWindow()
{

}
