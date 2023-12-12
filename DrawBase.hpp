#pragma once

#include <string>
#include "Config.hpp"

// DrawBase refers to stuffs that can be drawn on screen.
class DrawBase
{
public:
    CLASS_PTRS(DrawBase)

    virtual void draw() = 0;

    DrawBase(const DrawBase &) = delete; // there should not be a copy constructor available

protected:
    DrawBase(const std::string &name);

    std::string _objName;
    int _objId;
};
