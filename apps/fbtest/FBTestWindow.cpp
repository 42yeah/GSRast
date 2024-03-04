#include "FBTestWindow.hpp"
#include "Window.hpp"
#include "ColorfulShader.hpp"
#include "SamplerShader.hpp"
#include "BufferGeo.hpp"
#include "Framebuffer.hpp"

float tri[] = {
    0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f
};

FBTestWindow::FBTestWindow() : Window("FBTestWindow", DEFAULT_WINDOW_W, DEFAULT_WINDOW_H)
{
    configureFirstPersonCamera();

    ColorfulShader::Ptr colorfulShader = std::make_shared<ColorfulShader>();
    SamplerShader::Ptr samplerShader = std::make_shared<SamplerShader>(_camera);

    BufferGeo::Ptr triangle = std::make_shared<BufferGeo>();
    triangle->configure(tri, 3, sizeof(tri), colorfulShader);

    _framebuffer = std::make_shared<Framebuffer>(200, 100, true, samplerShader);
    _framebuffer->clear({ 1.0f, 0.0f, 1.0f, 1.0f });
    _framebuffer->addDrawable(triangle);

    addDrawable(_framebuffer);
}

void FBTestWindow::drawFrame()
{
    // Before window::drawFrame, we first draw OUR inner frame first
    _framebuffer->drawFrame();
    Window::drawFrame();
}
