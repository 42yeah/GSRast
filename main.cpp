// GSRast: Rasterize gaussian splats (based on the GS paper.)
#include <iostream>
#include "GSRasterWindow.hpp"

int main()
{
    GSRasterWindow window;
    while (!window.closed())
    {
        window.pollEvents();
        window.clear({ 1.0f, 0.5f, 0.0f, 1.0f });
        window.drawFrame();
        window.swapBuffers();
    }
    return 0;
}
