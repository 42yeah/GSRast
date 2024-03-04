#include "Config.hpp"

GlobalStacks _stacks;

std::ostream &operator<<(std::ostream &os, const glm::vec2 &p)
{
    return os << "(" << p.x << ", " << p.y << ")";
}

std::ostream &operator<<(std::ostream &os, const glm::vec3 &p)
{
    return os << "(" << p.x << ", " << p.y << ", " << p.z << ")";
}

std::ostream &operator<<(std::ostream &os, const glm::vec4 &p)
{
    return os << "(" << p.x << ", " << p.y << ", " << p.z << ", " << p.w << " | LEN=" << glm::length(p) << ")";
}

std::ostream &operator<<(std::ostream &os, const glm::mat4 &m)
{
    os << "[ ";
    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 4; col++)
        {
            os << m[col][row] << " ";
        }
        if (row != 3)
            os << std::endl << "  ";
    }
    return os << "]" << std::endl;
}

GlobalStacks &stacks()
{
    return _stacks;
}
