#pragma once

#include "BufferGeo.hpp"
#include "Config.hpp"
#include "ShaderBase.hpp"
#include "SplatData.hpp"


/**
 * I provide an interface of visualizing Gaussian Splats proper,
 * and I also take responsibility of actually loading them into
 * OpenGL buffers. (as SplatData uses SoA, I need to translate that as well.)
 */
class GSPointCloud : public BufferGeo
{
public:
    CLASS_PTRS(GSPointCloud)

    virtual bool configureFromSplatData(const SplatData::Ptr &splatData,
                                        const ShaderBase::Ptr &shader);

    virtual void draw() override;
    const glm::vec3 &getCenter() const;

    GSPointCloud();
    virtual ~GSPointCloud();

protected:
    glm::vec3 _center;
    SplatData::Ptr _splatData;
};
