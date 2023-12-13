#pragma once

#include "CameraBase.hpp"
#include <glm/glm.hpp>

class FirstPersonCamera : public CameraBase
{
public:
    CLASS_PTRS(FirstPersonCamera)

    /**
     * By default, when FirstPersonCamera was constructed,
     * it is situated at (0, 0, -1) looking at (0, 0, 1).
     */
    FirstPersonCamera();

    virtual const glm::mat4 &getView() const override;
    virtual const glm::mat4 &getPerspective() const override;
    virtual void update(const WindowBase &window) override;

    virtual void applyMotion(glm::vec3 dir);
    virtual void applyDelta(float dYaw, float dPitch);
    virtual void setSensitivity(float sensitivity);
    virtual const glm::vec3 &getYPR() const;
    virtual void setYPR(const glm::vec3 &ypr);
    virtual float getSpeed() const;
    virtual void setSpeed(float speed);
    virtual glm::vec3 getFront() const;
    virtual glm::vec3 getRight() const;
    virtual glm::vec3 getPosition() const;
    virtual void setPosition(const glm::vec3 &pos);
    void lookAt(const glm::vec3 &point);

protected:
    glm::vec3 _position, _front, _right;
    glm::mat4 _view, _perspective;
    glm::vec3 _ypr; // yaw pitch roll
    float _sensitivity, _speed;
};
