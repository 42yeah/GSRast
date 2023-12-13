#include "FirstPersonCamera.hpp"
#include "Config.hpp"
#include "WindowBase.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <iostream>


FirstPersonCamera::FirstPersonCamera() : _position(0.0f, 0.0f, -1.0f),
    _front(0.0f, 0.0f, 1.0f), _right(1.0f, 0.0f, 0.0f),
    _view(1.0f), _perspective(1.0f), _ypr(0.0f, 0.0f, 0.0f),
    _sensitivity(0.01f), _speed(1.0f)
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
    _front = glm::vec3(cosf(_ypr.y) * sinf(_ypr.x),
                       sinf(_ypr.y),
                       cosf(_ypr.y) * cosf(_ypr.x));
    _right = glm::normalize(glm::cross(_front, glm::vec3(0.0f, 1.0f, 0.0f)));
    _view = glm::lookAt(_position, _position + _front, glm::vec3(0.0f, 1.0f, 0.0f));
    float aspect = (float) window.getWidth() / window.getHeight();
    _perspective = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 100.0f);
}

void FirstPersonCamera::applyMotion(glm::vec3 dir)
{
    _position += dir * _speed;
}

void FirstPersonCamera::applyDelta(float dYaw, float dPitch)
{
    _ypr.x -= dYaw * _sensitivity;
    _ypr.y = glm::min(glm::max(_ypr.y - dPitch * _sensitivity, -0.5f * PI_F + EPSILON), 0.5f * PI_F - EPSILON);
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

glm::vec3 FirstPersonCamera::getFront() const
{
    return _front;
}

glm::vec3 FirstPersonCamera::getRight() const
{
    return _right;
}

glm::vec3 FirstPersonCamera::getPosition() const
{
    return _position;
}

void FirstPersonCamera::setPosition(const glm::vec3 &pos)
{
    _position = pos;
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
