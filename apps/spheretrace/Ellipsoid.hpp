#pragma once

#include "BufferGeo.hpp"
#include "ShaderBase.hpp"
#include "Config.hpp"
#include <glm/glm.hpp>


class Ellipsoid : public BufferGeo
{
public:
    CLASS_PTRS(Ellipsoid)

    Ellipsoid(ShaderBase::Ptr shader);
    ~Ellipsoid();

    const glm::vec3 &getCenter() const;
    void setCenter(const glm::vec3 &center);

    const glm::vec3 &getScale() const;
    void setScale(const glm::vec3 &scale);

    const glm::vec2 &getYawPitch() const;
    void setYawPitch(const glm::vec2 &yp);
    glm::mat3 getRotationMatrix() const;

protected:
    void compositeTransformations();

    glm::vec3 _center;
    glm::vec3 _scale;
    glm::vec2 _yp;
    glm::mat4 _rotation;
};
