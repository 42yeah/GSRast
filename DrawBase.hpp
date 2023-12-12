#pragma once

#include <string>
#include <glm/glm.hpp>
#include "Config.hpp"

// DrawBase refers to stuffs that can be drawn on screen.
class DrawBase
{
public:
    CLASS_PTRS(DrawBase)

    virtual void draw() = 0;

    DrawBase(const DrawBase &) = delete; // there should not be a copy constructor available

    void setModelMatrix(const glm::mat4 &mat);
    const glm::mat4 &getModelMatrix() const;

protected:
    DrawBase(const std::string &name);

    std::string _objName;
    int _objId;
    glm::mat4 _model;
};
