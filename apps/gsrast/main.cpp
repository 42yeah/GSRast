#include <iostream>
#include "GSRastWindow.hpp"

int main()
{
    GSRastWindow window;
    while (!window.closed())
    {
        window.pollEvents();
        window.clear({ 0.1f, 0.1f, 0.1f, 1.0f });
        window.drawFrame();
        window.swapBuffers();
    }
    return 0;
}
