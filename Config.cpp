#include "Config.hpp"


std::ostream &operator<<(std::ostream &os, const glm::vec3 &p)
{
    return os << "(" << p.x << ", " << p.y << ", " << p.z << ")";
}
