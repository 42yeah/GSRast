#include "Ellipsoid.hpp"
#include "ShaderBase.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>


float cube[] = {
    // Back
    -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f,
    1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f,
    1.0f, 1.0f, -1.0f, 0.0f, 0.0f, -1.0f,
    1.0f, 1.0f, -1.0f, 0.0f, 0.0f, -1.0f,
    -1.0f, 1.0f, -1.0f, 0.0f, 0.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f,

    // Front
    -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
    -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f,

    // Bottom
    -1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f,
    1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f,
    1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 0.0f,
    1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 0.0f,
    -1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 0.0f,
    -1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f,

    // Top
    -1.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    1.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
    1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
    -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
    -1.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,

    // Left
    -1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f,
    -1.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 0.0f, 0.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 0.0f, 0.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 0.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f,

    // Right
    1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f,
    1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f
};

Ellipsoid::Ellipsoid(ShaderBase::Ptr shader) : BufferGeo()
{
    _numVerts = 36;
    _shader = shader;

    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);

    glBufferData(GL_ARRAY_BUFFER, sizeof(cube), cube, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (void *) (sizeof(float) * 3));

    _center = glm::vec3(0.0f);
    _scale = glm::vec3(1.0f);
    _yp = glm::vec2(0.0f);
    _rotation = glm::mat4(1.0f);
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
    _model = _model * _rotation;
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


const glm::vec2 &Ellipsoid::getYawPitch() const
{
    return _yp;
}

void Ellipsoid::setYawPitch(const glm::vec2 &yp)
{
    _yp = yp;
    // First, rotate along the Y axis; then pitch along the Z axis
    _rotation = glm::mat4(cosf(_yp.y), -sinf(_yp.y), 0.0f, 0.0f,
                          sinf(_yp.y), cosf(_yp.y), 0.0f, 0.0f,
                          0.0f, 0.0f, 1.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 1.0f) *
                glm::mat4(cosf(_yp.x), 0.0f, -sinf(_yp.x), 0.0f,
                          0.0f, 1.0f, 0.0f, 0.0f,
                          sinf(_yp.x), 0.0f, cosf(_yp.x), 0.0f,
                          0.0f, 0.0f, 0.0f, 1.0f);
}

glm::mat3 Ellipsoid::getRotationMatrix() const
{
    return glm::mat3(_rotation);
}
