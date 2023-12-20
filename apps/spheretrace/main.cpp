#include <iostream>
#include "SphereTraceWindow.hpp"


int main()
{
    SphereTraceWindow window;
    while (!window.closed())
    {
        window.pollEvents();
        window.clear({ 0.2f, 0.1f, 0.2f, 1.0f });
        window.drawFrame();
        window.swapBuffers();
    }
    return 0;
}

