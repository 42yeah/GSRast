#include "BBox.hpp"
#include <limits>


BBox::BBox()
{
    reset();
}

void BBox::enclose(const glm::vec3 &p)
{
    infinite = false;
    min = glm::min(min, p);
    max = glm::max(max, p);
    center = 0.5f * (min + max);
}

bool BBox::inside(const glm::vec3 &p) const
{
    return infinite || (p.x > min.x && p.y > min.y && p.z > min.z &&
        p.x < max.x && p.y < max.y && p.z < max.z);
}

glm::vec3 BBox::span() const
{
    return max - min;
}

void BBox::reset()
{
    min = glm::vec3(std::numeric_limits<float>::max());
    max = glm::vec3(std::numeric_limits<float>::lowest());
    center = glm::vec3(0.0f);
    infinite = true;
}
