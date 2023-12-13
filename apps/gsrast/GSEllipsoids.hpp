#pragma once

#include "Config.hpp"
#include "GSPointCloud.hpp"


class GSEllipsoids : public GSPointCloud
{
public:
    CLASS_PTRS(GSEllipsoids)

    virtual bool configureFromPly(const std::string &path, ShaderBase::Ptr shader) override;

    virtual void draw() override;

    GSEllipsoids();
    ~GSEllipsoids();

protected:
    int _numInstances;
};
