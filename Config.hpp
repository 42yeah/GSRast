#pragma once

#define DEFAULT_WINDOW_W 800
#define DEFAULT_WINDOW_H 600
#define WINDOW_TITLE "GSRaster"

#include <memory>

#define CLASS_PTRS(CLASS_NAME) \
    using Ptr = std::shared_ptr<CLASS_NAME>;
