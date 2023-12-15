#include "Ellipsoid.hpp"
#include "ShaderBase.hpp"
#include <glm/gtc/matrix_transform.hpp>


float cube[] = {
    // Back
    -1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,

    // Front
    -1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f,

    // Bottom
    -1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f,

    // Top
    -1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f,

    // Left
    -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f,

    // Right
    1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, -1.0f,
};

Ellipsoid::Ellipsoid(ShaderBase::Ptr shader) : BufferGeo()
{
    configure(cube, 36, sizeof(cube), shader);
    _center = glm::vec3(0.0f);
    _scale = glm::vec3(1.0f);
}

Ellipsoid::~Ellipsoid()
{

}


const glm::vec3 &Ellipsoid::getCenter() const
{
    return _center;
}

void Ellipsoid::setCenter(const glm::vec3 &center)
{
    _center = center;
    compositeTransformations();
}

void Ellipsoid::compositeTransformations()
{
    // We need to composite the transformations together
    _model = glm::mat4(1.0f);
    _model = glm::translate(_model, _center);
    _model = glm::scale(_model, _scale);
}

const glm::vec3 &Ellipsoid::getScale() const
{
    return _scale;
}

void Ellipsoid::setScale(const glm::vec3 &scale)
{
    _scale = scale;
    compositeTransformations();
}
