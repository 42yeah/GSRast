#pragma once

#include "Config.hpp"
#include "GSPointCloud.hpp"


class GSEllipsoids : public GSPointCloud
{
public:
    CLASS_PTRS(GSEllipsoids)

    virtual bool configureFromPly(const std::string &path, ShaderBase::Ptr shader) override;

    virtual void draw() override;
    int getNumInstances() const;

    GSEllipsoids();
    ~GSEllipsoids();

    static GLuint generatePointsSSBO(const std::vector<glm::vec4> &points);

protected:
    int _numInstances;
    GLuint _positionSSBO;
    GLuint _scaleSSBO;
};
