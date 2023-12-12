#include "GSRastWindow.hpp"
#include "BufferGeo.hpp"
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

    float rect[] = {
        -1.0f, -1.0f, -1.0f,
        -0.2f, -0.3f, -0.4f,
        0.5f, 0.1f, 0.2f
    };
    BufferGeo::Ptr buf2 = std::make_shared<BufferGeo>();
    buf2->configure(rect, 3, sizeof(rect), _orbitalShader);

    addDrawable(bufGeo);
    addDrawable(buf2);
}

GSRastWindow::~GSRastWindow()
{

}
