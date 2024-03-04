#pragma once

#include "Config.hpp"
#include "GSPointCloud.hpp"
#include "ShaderBase.hpp"
#include "apps/gsrast/SplatData.hpp"
#include <glm/glm.hpp>


class GSEllipsoids : public GSPointCloud
{
public:
    CLASS_PTRS(GSEllipsoids)

    virtual bool configureFromSplatData(const SplatData::Ptr &splatData,
                                        const ShaderBase::Ptr &shader) override;

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
    SplatData::Ptr _splatData;
};
