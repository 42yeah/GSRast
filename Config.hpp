#pragma once

#define DEFAULT_WINDOW_W 800
#define DEFAULT_WINDOW_H 600
#define WINDOW_TITLE "GSRaster"

#include <memory>

#define CLASS_PTRS(CLASS_NAME) \
    using Ptr = std::shared_ptr<CLASS_NAME>;

// I recited them from my mind!
#define PI_F 3.1415f
#define PI 3.14159265
#define EPSILON 0.001f
