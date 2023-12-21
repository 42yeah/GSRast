#include <iostream>
#include "Config.hpp"
#include "FBTestWindow.hpp"


int main()
{
    FBTestWindow fbTestWindow;
    while (!fbTestWindow.closed())
    {
        fbTestWindow.pollEvents();
        fbTestWindow.clear({ 0.1f, 0.0f, 0.1f, 1.0f });
        fbTestWindow.drawFrame();
        fbTestWindow.swapBuffers();
    }
    return 0;
}
