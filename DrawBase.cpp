#include "DrawBase.hpp"
#include <map>
#include <string>

std::map<std::string, int> idCounters;

DrawBase::DrawBase(const std::string &name) : _model(1.0f)
{
    auto idx = idCounters.find(name);
    if (idx == idCounters.end())
    {
        idCounters[name] = 0;
        _objId = 0;
    }
    else
    {
        idx->second++;
        _objId = idx->second;
    }
    _objName = name;
}

void DrawBase::setModelMatrix(const glm::mat4 &mat)
{
    _model = mat;
}

const glm::mat4 &DrawBase::getModelMatrix() const
{
    return _model;
}
