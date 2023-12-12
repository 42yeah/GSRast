#include "DrawBase.hpp"
#include <map>
#include <string>

std::map<std::string, int> idCounters;

DrawBase::DrawBase(const std::string &name)
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
