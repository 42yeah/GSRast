#include "SphereTraceWindow.hpp"
#include "Config.hpp"
#include "Window.hpp"
#include "Cube.hpp"
#include "SphereTraceShader.hpp"
#include <memory>


SphereTraceWindow::SphereTraceWindow() : Window("Sphere trace", DEFAULT_WINDOW_W, DEFAULT_WINDOW_H)
{
    configureFirstPersonCamera();
    _firstPersonCamera->setPosition(glm::vec3(0.0f, 0.0f, -3.0f));
    _firstPersonCamera->setSpeed(3.0f);

    SphereTraceShader::Ptr shader = std::make_shared<SphereTraceShader>(getCamera());
    Cube::Ptr cube = std::make_shared<Cube>(shader);
    addDrawable(cube);

    _sphereCenter = glm::vec3(0.0f);
}

SphereTraceWindow::~SphereTraceWindow()
{

}
