#pragma once

#include "Config.hpp"
#include "GSPointCloud.hpp"
#include <glm/glm.hpp>


class GSEllipsoids : public GSPointCloud
{
public:
    CLASS_PTRS(GSEllipsoids)

    virtual bool configureFromPly(const std::string &path, ShaderBase::Ptr shader) override;

    virtual void draw() override;
    int getNumInstances() const;

    GSEllipsoids();
    ~GSEllipsoids();

protected:
    int _numInstances;
    GLuint _positionSSBO;
    GLuint _scaleSSBO;
    GLuint _colorSSBO;
    GLuint _quatSSBO;
    GLuint _alphaSSBO;
};
