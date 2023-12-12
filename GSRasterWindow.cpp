#include "GSRasterWindow.hpp"
#include "BufferGeo.hpp"
#include "Window.hpp"
#include "SimpleShader.hpp"


GSRasterWindow::GSRasterWindow() : Window(WINDOW_TITLE, DEFAULT_WINDOW_W, DEFAULT_WINDOW_H)
{
    SimpleShader::Ptr simpleShader = std::make_shared<SimpleShader>();

    float tri[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };
    BufferGeo::Ptr triangle = std::make_shared<BufferGeo>();
    triangle->configure(tri, 3, sizeof(tri), simpleShader);

    addDrawable(triangle);
}

GSRasterWindow::~GSRasterWindow()
{

}
