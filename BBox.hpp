#pragma once

#include <glm/glm.hpp>
#include "Config.hpp"

struct BBox
{
    CLASS_PTRS(BBox)

    BBox();

    void enclose(const glm::vec3 &p);
    bool inside(const glm::vec3 &p) const;
    glm::vec3 span() const;
    void reset();

    glm::vec3 min, max, center;
    bool infinite;
};
