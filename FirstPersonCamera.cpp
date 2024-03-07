#include "FirstPersonCamera.hpp"
#include "Config.hpp"
#include "WindowBase.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>


FirstPersonCamera::FirstPersonCamera() : _position(0.0f, 0.0f, -1.0f),
    _front(0.0f, 0.0f, 1.0f), _right(1.0f, 0.0f, 0.0f),
    _view(1.0f), _perspective(1.0f), _ypr(0.0f, 0.0f, 0.0f),
    _sensitivity(0.01f), _speed(1.0f), _near(DEFAULT_NEAR), _far(DEFAULT_FAR), _fov(DEFAULT_FOV),
    _invertUp(false)
{

}

const glm::mat4 &FirstPersonCamera::getView() const
{
    return _view;
}

const glm::mat4 &FirstPersonCamera::getPerspective() const
{
    return _perspective;
}

void FirstPersonCamera::update(const WindowBase &window)
{
    glm::vec3 arbitraryUp = !_invertUp ? glm::vec3(sinf(0.0f), cosf(0.0f), sinf(0.0f)) : glm::vec3(-sinf(0.0f), -cosf(0.0f), -sinf(0.0f));
    _front = glm::vec3(cosf(_ypr.y) * sinf(_ypr.x),
                       sinf(_ypr.y),
                       cosf(_ypr.y) * cosf(_ypr.x));
    _right = glm::normalize(glm::cross(_front, arbitraryUp));
    _view = glm::lookAt(_position, _position + _front, arbitraryUp);
    float aspect = (float) window.getWidth() / window.getHeight();
    _perspective = glm::perspective(_fov, aspect, _near, _far);
}

void FirstPersonCamera::applyMotion(glm::vec3 dir)
{
    _position += dir * _speed;
}

void FirstPersonCamera::applyDelta(float dYaw, float dPitch, float dRoll)
{
    float modifier = _invertUp ? -1.0f : 1.0f;
    _ypr.x -= modifier * dYaw * _sensitivity;
    _ypr.y = glm::min(glm::max(_ypr.y - modifier * dPitch * _sensitivity, -0.5f * PI_F + EPSILON), 0.5f * PI_F - EPSILON);
    _ypr.z += dRoll;
}

void FirstPersonCamera::setSensitivity(float sensitivity)
{
    _sensitivity = sensitivity;
}

const glm::vec3 &FirstPersonCamera::getYPR() const
{
    return _ypr;
}

void FirstPersonCamera::setYPR(const glm::vec3 &ypr)
{
    _ypr = ypr;
}

float FirstPersonCamera::getSpeed() const
{
    return _speed;
}

void FirstPersonCamera::setSpeed(float speed)
{
    _speed = speed;
}

const glm::vec3 &FirstPersonCamera::getFront() const
{
    return _front;
}

const glm::vec3 &FirstPersonCamera::getRight() const
{
    return _right;
}

const glm::vec3 &FirstPersonCamera::getPosition() const
{
    return _position;
}

void FirstPersonCamera::setPosition(const glm::vec3 &pos)
{
    _position = pos;
}

glm::vec2 FirstPersonCamera::getNearFar() const
{
    return glm::vec2(_near, _far);
}

void FirstPersonCamera::lookAt(const glm::vec3 &point)
{
    glm::vec3 dir = glm::normalize(point - _position);
    _ypr.y = asinf(dir.y);
    float yaw1 = asinf(dir.x / cosf(_ypr.y));
    float yaw2 = acosf(dir.z / cosf(_ypr.y));
    if (fabs(yaw1 - yaw2) > EPSILON)
    {
        _ypr.x += PI_F;
    }
    // We don't support rolls.
    _ypr.z = 0.0f;
}

void FirstPersonCamera::setNearFar(float near, float far)
{
    _near = near;
    _far = far;
}

float FirstPersonCamera::getFOV() const
{
    return _fov;
}

float FirstPersonCamera::getSensitivity() const
{
    return _sensitivity;
}

void FirstPersonCamera::setFOV(float fov)
{
    _fov = fov;
}

void FirstPersonCamera::setInvertUp(bool invert)
{
    _invertUp = invert;
}

bool FirstPersonCamera::getInvertUp() const
{
    return _invertUp;
}
